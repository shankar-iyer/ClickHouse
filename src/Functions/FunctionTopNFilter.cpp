#include <memory>
#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/FunctionFactory.h>
#include <Functions/IFunction.h>
#include <Interpreters/Context.h>
#include <IO/WriteHelpers.h>
#include <Common/CurrentThread.h>
#include <Common/logger_useful.h>
#include <Processors/TopNThresholdTracker.h>
#include <Interpreters/convertFieldToType.h>
#include <Functions/FunctionFactory.h>
#include <Functions/IFunctionAdaptors.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int TOO_FEW_ARGUMENTS_FOR_FUNCTION;
}

class FunctionTopNFilter: public IFunction
{
public:
    static constexpr auto name = "__topNFilter";
    /// static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionTopNFilter>(); }

    explicit FunctionTopNFilter(TopNThresholdTrackerPtr threshold_tracker_)
        : threshold_tracker(threshold_tracker_)
    {
    }

    String getName() const override
    {
        return name;
    }

    bool isVariadic() const override { return false; }
    bool isInjective(const ColumnsWithTypeAndName &) const override { return false; }
    bool isSuitableForConstantFolding() const override { return false; }
    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return false; }
    size_t getNumberOfArguments() const override { return 1; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (arguments.size() != 1)
            throw Exception(ErrorCodes::TOO_FEW_ARGUMENTS_FOR_FUNCTION,
                            "Number of arguments for function {} can't be {}, should be 1",
                            getName(), arguments.size());

        return std::make_shared<DataTypeUInt8>();
    }

    DataTypePtr getReturnTypeForDefaultImplementationForDynamic() const override
    {
        return std::make_shared<DataTypeUInt8>();
    }

    bool useDefaultImplementationForConstants() const override { return true; }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        auto data_column = arguments[0].column;
#if 0
	LOG_TRACE(getLogger(""), "TopN::executeImpl {}", input_rows_count);
	if (data_column->size() > 0)
	{
	LOG_TRACE(getLogger(""), "Inside TopNFilter {} {} {} {}", data_column->getValueNameAndType(0).first, data_column->getValueNameAndType(0).second->getName(), data_column->get64(0), data_column->getInt(0));
	}
#endif
        if (threshold_tracker && threshold_tracker->isSet())
        {
            auto current_threshold = threshold_tracker->get();
	    auto data_type = arguments[0].type;
            ColumnPtr threshold_column = data_type->createColumnConst(input_rows_count, convertFieldToType(current_threshold, *data_type));

	    auto context = Context::getGlobalContextInstance();
	    auto compare = FunctionFactory::instance().get("greater", context);
	    auto left = arguments[0];
	    ColumnWithTypeAndName right{threshold_column, data_type, {}};
	    auto elem_compare = compare->build(ColumnsWithTypeAndName{left, right});
	    return elem_compare->execute({left, right}, elem_compare->getResultType(), input_rows_count, /* dry_run = */ false);
        }
	else
            return DataTypeUInt8().createColumnConst(input_rows_count, true);
    }
private:
    TopNThresholdTrackerPtr threshold_tracker;
};

#if 0
REGISTER_FUNCTION(TopNFilter)
{
    FunctionDocumentation::Description description = R"(Special function for TopN filtering.)";
    FunctionDocumentation::Syntax syntax = "__filterContains(filter_name, key)";
    FunctionDocumentation::Arguments arguments = {
        {"filter_name", "Internal name of runtime filter. It is built by BuildRuntimeFilterStep.", {"String"}},
        {"key", "Value of any type that is checked to be present in the filter", {}}
    };
    FunctionDocumentation::ReturnedValue returned_value = {"True if the key was found in the filter", {"Bool"}};
    FunctionDocumentation::Examples examples = {{"Example", "This function is not supposed to be used in user queries. It might be added to query plan during optimization. ", ""}};
    FunctionDocumentation::IntroducedIn introduced_in = {25, 10};
    FunctionDocumentation::Category category = FunctionDocumentation::Category::Other;

    factory.registerFunction<FunctionTopNFilter>({description, syntax, arguments, returned_value, examples, introduced_in, category}, FunctionFactory::Case::Sensitive);
}
#endif

FunctionOverloadResolverPtr createInternalFunctionTopNFilterResolver(TopNThresholdTrackerPtr threshold_tracker_)
{
    return std::make_shared<FunctionToOverloadResolverAdaptor>(std::make_shared<FunctionTopNFilter>(threshold_tracker_));
};

}
