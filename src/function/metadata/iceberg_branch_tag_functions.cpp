#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/planner/binder.hpp"

#include "catalog/rest/catalog_entry/table/iceberg_table.hpp"
#include "catalog/rest/catalog_entry/table/iceberg_table_schema_version.hpp"
#include "catalog/rest/transaction/iceberg_transaction.hpp"
#include "catalog/rest/transaction/iceberg_transaction_data.hpp"
#include "common/iceberg_utils.hpp"
#include "core/metadata/iceberg_table_metadata.hpp"
#include "function/iceberg_functions.hpp"

namespace duckdb {

namespace {

enum class RefOperation : uint8_t { CREATE, REPLACE, RENAME, DROP };

struct RefFunctionInfo : public TableFunctionInfo {
	RefFunctionInfo(string function_name_p, RefOperation operation_p, bool is_branch_p)
	    : function_name(std::move(function_name_p)), operation(operation_p), is_branch(is_branch_p) {
	}

	string function_name;
	RefOperation operation;
	//! Whether the function works on branches or on tags
	bool is_branch;
};

struct RefRetention {
	optional<int32_t> min_snapshots_to_keep;
	optional<int64_t> max_snapshot_age_ms;
	optional<int64_t> max_ref_age_ms;
};

struct RefFunctionBindData : public TableFunctionData {
	optional_ptr<IcebergTableSchemaVersion> iceberg_table;
	optional_ptr<const RefFunctionInfo> info;
	string ref_name;
	//! The new name of a renamed branch
	string new_ref_name;
	//! The snapshot to point at; main's current snapshot if not given
	optional<int64_t> snapshot_id;
	RefRetention retention;
};

struct RefFunctionGlobalState : public GlobalTableFunctionState {
	static unique_ptr<GlobalTableFunctionState> Init(ClientContext &, TableFunctionInitInput &) {
		return make_uniq<RefFunctionGlobalState>();
	}

	bool finished = false;
};

//! The ref a function set or removed
struct RefResult {
	string name;
	string type;
	int64_t snapshot_id = 0;
};

static string RefTypeName(bool is_branch) {
	return is_branch ? "branch" : "tag";
}

static bool CheckTableIsIcebergTable(optional_ptr<CatalogEntry> entry) {
	auto &catalog_entry = entry->Cast<InCatalogEntry>();
	auto &catalog = catalog_entry.catalog;
	if (catalog.GetCatalogType() != "iceberg") {
		return false;
	}
	if (entry->type != CatalogType::TABLE_ENTRY) {
		return false;
	}
	return true;
}

static void VerifyInputIsNotAFile(ClientContext &context, string &input_string, const string &function_name) {
	auto qualified_name = QualifiedName::ParseComponents(input_string);
	if (qualified_name.size() == 1) {
		auto &fs = FileSystem::GetFileSystem(context);
		if (fs.DirectoryExists(input_string) || fs.FileExists(input_string)) {
			throw InvalidInputException("Cannot call %s() on a file/directory", function_name);
		}
	}
}

static string GetRefName(const Value &value, const string &function_name, const string &parameter) {
	if (value.IsNull()) {
		throw InvalidInputException("%s: %s cannot be NULL", function_name, parameter);
	}
	auto name = StringValue::Get(value);
	if (name.empty()) {
		throw InvalidInputException("%s: Branch or tag name cannot be empty", function_name);
	}
	return name;
}

static RefRetention GetRetention(const named_argument_map_t &named_parameters, const string &function_name) {
	RefRetention result;
	for (auto &entry : named_parameters) {
		auto option = StringUtil::Lower(entry.first.GetIdentifierName());
		auto &value = entry.second;
		if (value.IsNull()) {
			throw InvalidInputException("%s: %s cannot be NULL", function_name, option);
		}
		auto number = value.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
		//! Same rule as Iceberg's SnapshotRef: every retention setting must be positive
		if (number <= 0) {
			throw InvalidInputException("%s: %s must be greater than 0", function_name, option);
		}
		if (option == "min_snapshots_to_keep") {
			result.min_snapshots_to_keep = value.GetValue<int32_t>();
		} else if (option == "max_snapshot_age_ms") {
			result.max_snapshot_age_ms = number;
		} else if (option == "max_ref_age_ms") {
			result.max_ref_age_ms = number;
		}
	}
	return result;
}

static unique_ptr<FunctionData> RefFunctionBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<Identifier> &names) {
	auto &info = input.info->Cast<RefFunctionInfo>();
	auto result = make_uniq<RefFunctionBindData>();
	result->info = &info;

	if (input.inputs[0].IsNull()) {
		throw InvalidInputException("%s: table_name cannot be NULL", info.function_name);
	}
	auto input_string = input.inputs[0].ToString();
	VerifyInputIsNotAFile(context, input_string, info.function_name);
	auto iceberg_table = IcebergUtils::GetTableEntry(context, input_string);
	if (!CheckTableIsIcebergTable(iceberg_table)) {
		throw InvalidInputException("Cannot call %s on non-iceberg table", info.function_name);
	}
	result->iceberg_table = iceberg_table->Cast<IcebergTableSchemaVersion>();

	result->ref_name = GetRefName(input.inputs[1], info.function_name, info.is_branch ? "branch_name" : "tag_name");
	if (info.operation == RefOperation::RENAME) {
		result->new_ref_name = GetRefName(input.inputs[2], info.function_name, "new_branch_name");
	} else if (input.inputs.size() > 2) {
		if (input.inputs[2].IsNull()) {
			throw InvalidInputException("%s: snapshot_id cannot be NULL", info.function_name);
		}
		result->snapshot_id = input.inputs[2].DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
	}
	result->retention = GetRetention(input.named_parameters, info.function_name);

	if (input.binder) {
		DatabaseModificationType modification;
		modification |= DatabaseModificationType::ALTER_TABLE;
		input.binder->GetStatementProperties().RegisterDBModify(iceberg_table->ParentCatalog(), context, modification);
	}

	names.emplace_back("name");
	return_types.push_back(LogicalType::VARCHAR);
	names.emplace_back("type");
	return_types.push_back(LogicalType::VARCHAR);
	names.emplace_back("snapshot_id");
	return_types.push_back(LogicalType::BIGINT);
	return std::move(result);
}

static optional_ptr<const rest_api_objects::SnapshotReference> FindRef(const IcebergTableMetadata &metadata,
                                                                       const string &name) {
	if (!metadata.refs) {
		return nullptr;
	}
	auto entry = metadata.refs->additional_properties.find(name);
	if (entry == metadata.refs->additional_properties.end()) {
		return nullptr;
	}
	return &entry->second;
}

//! Throws unless 'name' is an existing ref of the given type
static const rest_api_objects::SnapshotReference &GetExistingRef(const IcebergTableMetadata &metadata,
                                                                 const string &name, bool is_branch) {
	auto ref = FindRef(metadata, name);
	if (!ref) {
		throw InvalidInputException("%s \"%s\" does not exist", is_branch ? "Branch" : "Tag", name);
	}
	auto ref_is_branch = StringUtil::CIEquals(ref->type, "branch");
	if (ref_is_branch != is_branch) {
		throw InvalidInputException("\"%s\" is a %s, not a %s", name, RefTypeName(ref_is_branch),
		                            RefTypeName(is_branch));
	}
	return *ref;
}

static void VerifyRefDoesNotExist(const IcebergTableMetadata &metadata, const string &name) {
	if (FindRef(metadata, name)) {
		throw InvalidInputException("Branch or tag \"%s\" already exists", name);
	}
}

static int64_t GetTargetSnapshot(const IcebergTableMetadata &metadata, const RefFunctionBindData &bind_data) {
	auto &name = bind_data.ref_name;
	if (!bind_data.snapshot_id) {
		if (!metadata.current_snapshot_id) {
			throw InvalidInputException("Cannot create %s \"%s\": the table has no current snapshot",
			                            RefTypeName(bind_data.info->is_branch), name);
		}
		return *metadata.current_snapshot_id;
	}
	auto snapshot_id = *bind_data.snapshot_id;
	if (!metadata.FindSnapshotByIdInternal(snapshot_id)) {
		throw InvalidInputException("Cannot point \"%s\" at unknown snapshot %lld", name, snapshot_id);
	}
	return snapshot_id;
}

//! Given retention settings replace the ref's settings, the others are kept
static void ApplyRetention(rest_api_objects::SnapshotReference &ref, const RefRetention &retention) {
	if (retention.min_snapshots_to_keep) {
		ref.min_snapshots_to_keep = retention.min_snapshots_to_keep;
	}
	if (retention.max_snapshot_age_ms) {
		ref.max_snapshot_age_ms = retention.max_snapshot_age_ms;
	}
	if (retention.max_ref_age_ms) {
		ref.max_ref_age_ms = retention.max_ref_age_ms;
	}
}

static RefResult ApplyRefOperation(const RefFunctionBindData &bind_data, IcebergTableMetadata &metadata,
                                   IcebergTransactionData &transaction_data) {
	auto &info = *bind_data.info;
	auto &name = bind_data.ref_name;
	if (name == "main" || bind_data.new_ref_name == "main") {
		throw InvalidInputException("%s cannot change the main branch", info.function_name);
	}
	switch (info.operation) {
	case RefOperation::CREATE: {
		VerifyRefDoesNotExist(metadata, name);
		rest_api_objects::SnapshotReference ref;
		ref.type = RefTypeName(info.is_branch);
		ref.snapshot_id = GetTargetSnapshot(metadata, bind_data);
		ApplyRetention(ref, bind_data.retention);
		transaction_data.TableSetSnapshotRef(metadata, name, ref);
		return RefResult {name, ref.type, ref.snapshot_id};
	}
	case RefOperation::REPLACE: {
		auto ref = GetExistingRef(metadata, name, info.is_branch).Copy();
		ref.snapshot_id = GetTargetSnapshot(metadata, bind_data);
		ApplyRetention(ref, bind_data.retention);
		transaction_data.TableSetSnapshotRef(metadata, name, ref);
		return RefResult {name, ref.type, ref.snapshot_id};
	}
	case RefOperation::RENAME: {
		//! A rename sets the new name and removes the old one in the same commit
		auto ref = GetExistingRef(metadata, name, true).Copy();
		auto &new_name = bind_data.new_ref_name;
		VerifyRefDoesNotExist(metadata, new_name);
		transaction_data.TableSetSnapshotRef(metadata, new_name, ref);
		transaction_data.TableRemoveSnapshotRef(metadata, name);
		return RefResult {new_name, ref.type, ref.snapshot_id};
	}
	case RefOperation::DROP: {
		auto &ref = GetExistingRef(metadata, name, info.is_branch);
		RefResult result {name, ref.type, ref.snapshot_id};
		transaction_data.TableRemoveSnapshotRef(metadata, name);
		return result;
	}
	}
	throw InternalException("Unknown branch or tag operation");
}

static void RefFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<RefFunctionBindData>();
	auto &global_state = data.global_state->Cast<RefFunctionGlobalState>();
	if (!bind_data.iceberg_table || global_state.finished) {
		output.SetChildCardinality(0);
		return;
	}

	auto &table_info = bind_data.iceberg_table->table_info;
	auto &iceberg_transaction = IcebergTransaction::Get(context, bind_data.iceberg_table->catalog);
	RefResult result;
	ApplyTableUpdate(table_info, iceberg_transaction, [&](IcebergTable &table) {
		auto &transaction_data = table.GetOrCreateTransactionData(iceberg_transaction);
		result = ApplyRefOperation(bind_data, table.GetMetadataMutable(), transaction_data);
	});

	output.data[0].SetValue(0, Value(result.name));
	//! Upper case, matching iceberg_refs
	output.data[1].SetValue(0, Value(StringUtil::Upper(result.type)));
	output.data[2].SetValue(0, Value::BIGINT(result.snapshot_id));
	output.SetChildCardinality(1);
	global_state.finished = true;
}

struct RefFunctionArgument {
	const char *name;
	LogicalType type;
};

static TableFunction CreateRefFunction(const shared_ptr<RefFunctionInfo> &info,
                                       const vector<RefFunctionArgument> &arguments) {
	FunctionSignature signature;
	signature.AddPositionalOnly("table_name", LogicalType::VARCHAR);
	for (auto &argument : arguments) {
		signature.AddPositionalOnly(argument.name, argument.type);
	}
	TableFunction function(std::move(signature), RefFunction, RefFunctionBind, RefFunctionGlobalState::Init);
	auto operation = info->operation;
	if (operation == RefOperation::CREATE || operation == RefOperation::REPLACE) {
		auto is_branch = info->is_branch;
		function.GetSignature().WithTypedKwargs("options", [&](TypedKwargs &options) {
			if (is_branch) {
				options.Add("min_snapshots_to_keep", LogicalType::INTEGER)
				    .Add("max_snapshot_age_ms", LogicalType::BIGINT);
			}
			options.Add("max_ref_age_ms", LogicalType::BIGINT);
		});
	}
	function.function_info = info;
	return function;
}

static TableFunctionSet CreateRefFunctionSet(const string &function_name, RefOperation operation, bool is_branch) {
	auto info = make_shared_ptr<RefFunctionInfo>(function_name, operation, is_branch);
	const char *name_argument = is_branch ? "branch_name" : "tag_name";
	TableFunctionSet function_set {Identifier(function_name)};
	if (operation == RefOperation::CREATE || operation == RefOperation::DROP) {
		function_set.AddFunction(CreateRefFunction(info, {{name_argument, LogicalType::VARCHAR}}));
	}
	if (operation == RefOperation::CREATE || operation == RefOperation::REPLACE) {
		//! iceberg_snapshots returns snapshot ids as UBIGINT, so take those without a cast
		for (auto id_type : {LogicalType::BIGINT, LogicalType::UBIGINT}) {
			function_set.AddFunction(
			    CreateRefFunction(info, {{name_argument, LogicalType::VARCHAR}, {"snapshot_id", id_type}}));
		}
	}
	if (operation == RefOperation::RENAME) {
		function_set.AddFunction(CreateRefFunction(
		    info, {{name_argument, LogicalType::VARCHAR}, {"new_branch_name", LogicalType::VARCHAR}}));
	}
	return function_set;
}

} // namespace

vector<TableFunctionSet> IcebergFunctions::GetIcebergBranchTagFunctions() {
	vector<TableFunctionSet> result;
	result.push_back(CreateRefFunctionSet("iceberg_create_branch", RefOperation::CREATE, true));
	result.push_back(CreateRefFunctionSet("iceberg_create_tag", RefOperation::CREATE, false));
	result.push_back(CreateRefFunctionSet("iceberg_replace_branch", RefOperation::REPLACE, true));
	result.push_back(CreateRefFunctionSet("iceberg_replace_tag", RefOperation::REPLACE, false));
	result.push_back(CreateRefFunctionSet("iceberg_rename_branch", RefOperation::RENAME, true));
	result.push_back(CreateRefFunctionSet("iceberg_drop_branch", RefOperation::DROP, true));
	result.push_back(CreateRefFunctionSet("iceberg_drop_tag", RefOperation::DROP, false));
	return result;
}

} // namespace duckdb
