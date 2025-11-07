#include <Columns/ColumnConst.h>
#include <Core/Field.h>
#include <Core/SortDescription.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/IFunction.h>
#include <Processors/QueryPlan/ExpressionStep.h>
#include <Processors/QueryPlan/FilterStep.h>
#include <Processors/QueryPlan/LimitStep.h>
#include <Processors/QueryPlan/Optimizations/Optimizations.h>
#include <Processors/QueryPlan/QueryPlan.h>
#include <Processors/QueryPlan/ReadFromMergeTree.h>
#include <Processors/QueryPlan/SortingStep.h>
#include <Common/logger_useful.h>
#include <Functions/FunctionFactory.h>

namespace DB::QueryPlanOptimizations
{

size_t tryPushDownOrderByLimit(QueryPlan::Node * parent_node, QueryPlan::Nodes & /* nodes*/, const Optimization::ExtraSettings & /*settings*/)
{
    QueryPlan::Node * node = parent_node;

    auto * limit_step = typeid_cast<LimitStep *>(node->step.get());
    if (!limit_step)
        return 0;

    if (node->children.size() != 1)
        return 0;
    node = node->children.front();
    auto * sorting_step = typeid_cast<SortingStep *>(node->step.get());
    if (!sorting_step)
        return 0;

    if (node->children.size() != 1)
        return 0;
    node = node->children.front();

    ExpressionStep * expression_step = typeid_cast<ExpressionStep *>(node->step.get());
    if (expression_step)
    {
        LOG_TRACE(getLogger(""), "expression_step is found");
        if (node->children.size() != 1)
            return 0;
        node = node->children.front();
    }
    FilterStep * filter_step = typeid_cast<FilterStep *>(node->step.get());
    if (filter_step)
    {
        LOG_TRACE(getLogger(""), "filter_step is found");
        if (node->children.size() != 1)
            return 0;
        node = node->children.front();
    }

    auto * read_from_mergetree_step = typeid_cast<ReadFromMergeTree *>(node->step.get());
    if (!read_from_mergetree_step)
        return 0;

    if (read_from_mergetree_step->getPrewhereInfo())
        LOG_TRACE(getLogger(""), "prewhere is found.");
    else
        LOG_TRACE(getLogger(""), "prewhere is nout found");

    /// Extract N
    size_t n = limit_step->getLimitForSorting();
    if (n > 10000) /// settings.max_limit_to_push_down_topn_predicate)
        return 0;

    SortingStep::Type sorting_step_type = sorting_step->getType();
    if (sorting_step_type != SortingStep::Type::Full)
        return 0;

    const auto & sort_description = sorting_step->getSortDescription();

    LOG_TRACE(getLogger(""), "optimizeOrderByLimit {} {}", n, sort_description.front().column_name);

    const auto & sort_column = sorting_step->getInputHeaders().front()->getByName(sort_description.front().column_name);
    if (!sort_column.type->isValueRepresentedByNumber())
        return 0;

/*
    const auto * sort_column_from_read
        = read_from_mergetree_step->getStorageMetadata()->getColumns().tryGet(sort_description.front().column_name_in_storage);
    if (!sort_column_from_read || !sort_column_from_read->type->equals(*sort_column.type))
        return 0;
*/

    auto sort_column_name = sort_description.front().column_name.substr(sort_description.front().column_name.find('.') + 1);
    ActionsDAG & actions_dag = expression_step->getExpression();
    LOG_TRACE(getLogger(""), "DAG before {}", actions_dag.dumpDAG());
    /// const auto & key_column_node = actions_dag.findInOutputs(sort_column_name);
#if 0
    const auto & key_column_node = actions_dag.findInOutputs(sort_description.front().column_name);

    const auto * filter_argument = &key_column_node;
    auto filter_function = FunctionFactory::instance().get("__topNFilter", /*query_context*/nullptr);
    auto & function_node = actions_dag.addFunction(filter_function, {filter_argument}, {});
    actions_dag.addOrReplaceInOutputs(function_node);
    LOG_TRACE(getLogger(""), "DAG after {}", actions_dag.dumpDAG());
#endif
    if (filter_step)
    {
    ActionsDAG & filter_dag = filter_step->getExpression();
    LOG_TRACE(getLogger(""), "Filter DAG is {}", filter_dag.dumpDAG());
    }
 #if 0
    auto new_prewhere_info = std::make_shared<PrewhereInfo>();
    const auto * column_input = &new_prewhere_info->prewhere_actions.addInput(sort_column_name, std::make_shared<DataTypeInt32>());
    const auto & alias1 = &new_prewhere_info->prewhere_actions.addAlias(*column_input, "_alias_v1");
    new_prewhere_info->prewhere_actions.getOutputs().push_back(alias1);
    LOG_TRACE(getLogger(""), "DAG just after {}", new_prewhere_info->prewhere_actions.dumpDAG());
    const auto & key_column_node = new_prewhere_info->prewhere_actions.findInOutputs("_alias_v1");
    ActionsDAG::NodeRawConstPtrs children = {&key_column_node};
    auto filter_function = FunctionFactory::instance().get("__topNFilter", /*query_context*/nullptr);
    const auto * function_node = &new_prewhere_info->prewhere_actions.addFunction(filter_function, children, {});
    new_prewhere_info->prewhere_actions.getOutputs().push_back(function_node);
    LOG_TRACE(getLogger(""), "DAG after {}", new_prewhere_info->prewhere_actions.dumpDAG());
 #endif
            NameAndTypePair top_n_column(sort_column_name, std::make_shared<DataTypeInt32>());
            auto prewhere_info = std::make_shared<PrewhereInfo>();
            prewhere_info->prewhere_actions = ActionsDAG({top_n_column});
            auto filter_function = FunctionFactory::instance().get("__topNFilter", /*query_context*/nullptr);
            const auto & prewhere_node = prewhere_info->prewhere_actions.addFunction(
                filter_function, {prewhere_info->prewhere_actions.getInputs().front()}, {});
            prewhere_info->prewhere_actions.getOutputs().push_back(&prewhere_node);
            prewhere_info->prewhere_column_name = prewhere_node.result_name;
            prewhere_info->remove_prewhere_column = true;
            prewhere_info->need_filter = true;
	    LOG_TRACE(getLogger(""), "New Prewhere {}", prewhere_info->prewhere_actions.dumpDAG());
            read_from_mergetree_step->updatePrewhereInfo(prewhere_info);


    read_from_mergetree_step->setTopNColumn({sort_description.front().column_name, sort_column.type, n});
    /// read_from_mergetree_step->prewhere_info = std::move(new_prewhere_info);
    /// read_from_mergetree_step->updatePrewhereInfo(new_prewhere_info);
    sorting_step->setTopNThresholdUpdate(true);

    return 0;
}

}
