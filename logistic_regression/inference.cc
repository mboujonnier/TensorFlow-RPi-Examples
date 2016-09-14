//
// The googlenet_graph.pb file included by default is created from Inception.

#include <fstream>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/logging.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

// Reads a model graph definition from disk, and creates a session object you can use to run it.
Status LoadGraph(string graph_file_name, std::unique_ptr<tensorflow::Session>* session)
{
	tensorflow::GraphDef graph_def;
	Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok())
	{
		LOG(ERROR) << load_graph_status << " with file " << graph_file_name;
		return load_graph_status;
	}

	session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
	Status session_create_status = (*session)->Create(graph_def);
	if (!session_create_status.ok())
	{
		return session_create_status;
	}

	LOG(INFO) << graph_def.DebugString();

	return Status::OK();
}

int main(int argc, char* argv[])
{
	string ograph = "/home/tas/ML/tensorflow/tensorflow/contrib/pi_examples/logistic_regression/data/output_graph.pb";

	string root_dir = "";
	std::vector<Tensor> outputs;

	// We need to call this to set up global state for TensorFlow.
	tensorflow::port::InitMain(argv[0], &argc, &argv);

	// First we load and initialize the model.
	std::unique_ptr<tensorflow::Session> session;
	string ograph_path = tensorflow::io::JoinPath(root_dir, ograph);

	Status load_ograph_status = LoadGraph(ograph_path, &session);
	if (!load_ograph_status.ok())
	{
		LOG(ERROR) << load_ograph_status;
		return -1;
	}
	else
	{
		LOG(INFO) << "Output Graph is loaded successfully";
	}

	// Create the test data to check accuracy
	LOG(INFO) << "Generate Test data";
	float test_X[3][3] = {{ 1.0, 2.0, 2.0}, {1.0, 5.0, 5.0}, {1.0, 4.0, 3.0}};
	tensorflow::Tensor inputs_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({3, 1}));
	auto inputs_tensor_mapped = inputs_tensor.tensor<float, 2>();


	LOG(INFO) << "Run the model";

	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3; j++)
		{
			inputs_tensor_mapped(j, 0) = test_X[i][j];
		}

		// virtual Status tensorflow::Session::Run(const std::vector< std::pair< string, Tensor > > &inputs, const std::vector< string > &output_tensor_names,	const std::vector< string > &target_node_names,	std::vector< Tensor > *outputs)=0
		Status run_status = session->Run({{"input_node/X", inputs_tensor}}, {{"output_hyp"}}, {}, &outputs);
		if (!run_status.ok())
		{
			LOG(ERROR) << "Running model failed: " << run_status;
			return -1;
		}

		auto output_hyp =	outputs[0].flat<float>();
		LOG(INFO) << "output_hyp: [[" << inputs_tensor_mapped(0, 0) << "],[" << inputs_tensor_mapped(1, 0) << "],[" << inputs_tensor_mapped(2, 0) << "]] => " << ((output_hyp(0) > 0.5)?"True":"False");
	}

	session->Close();

	return 0;
}
