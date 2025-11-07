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
    static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionTopNFilter>(); }

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
	LOG_TRACE(getLogger(""), "TopN::executeImpl {}", input_rows_count);
	if (data_column->size() > 0)
	{
	LOG_TRACE(getLogger(""), "Inside TopNFilter {} {} {} {}", data_column->getValueNameAndType(0).first, data_column->getValueNameAndType(0).second->getName(), data_column->get64(0), data_column->getInt(0));
	}
        return DataTypeUInt8().createColumnConst(input_rows_count, true);
    }
};

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

}
