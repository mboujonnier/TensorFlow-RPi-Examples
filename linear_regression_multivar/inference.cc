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
	string ograph = "/home/tas/ML/tensorflow/tensorflow/contrib/pi_examples/linear_regression_multivar/data/output_graph.pb";

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

	tensorflow::Tensor inputs_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({3, 1}));
	auto inputs_tensor_mapped = inputs_tensor.tensor<float, 2>();

  float surface = 1650.0;
  float bedrooms = 3.0;

  // get means and std deviations
	LOG(INFO) << "Retrieve means and std deviations from the model to normalize inputs";
  Status run_status = session->Run({}, {{"output_means"}, {"output_stds"}}, {}, &outputs);
  if (!run_status.ok())
	{
		LOG(ERROR) << "Running model failed: " << run_status;
		return -1;
	}

  auto output_means = 	outputs[0].flat<double>();
  auto output_std = 	outputs[1].flat<double>();
  LOG(INFO) << "means: [" << output_means(0) << ", " << output_means(1) << ", " << output_means(2) <<"]";
	LOG(INFO) << "std dev: [" << output_std(0) << ", " << output_std(1) << ", " << output_std(2) <<"]";

	LOG(INFO) << "Creating a test tensor and normalize it";
  inputs_tensor_mapped(0, 0) = 1.0;
  inputs_tensor_mapped(1, 0) = (surface - output_means(1))/output_std(1);
  inputs_tensor_mapped(2, 0) = (bedrooms - output_means(2))/output_std(2);

	LOG(INFO) << "Run the model";
	// virtual Status tensorflow::Session::Run(const std::vector< std::pair< string, Tensor > > &inputs, const std::vector< string > &output_tensor_names,	const std::vector< string > &target_node_names,	std::vector< Tensor > *outputs)=0
	run_status = session->Run({{"input_node/OneHouseX", inputs_tensor}}, {{"output_price"}}, {}, &outputs);
	if (!run_status.ok())
	{
		LOG(ERROR) << "Running model failed: " << run_status;
		return -1;
	}

	auto output_price= 	outputs[0].flat<float>();
	LOG(INFO) << "Estimated price for an house of " << surface << " sqfeet and " << bedrooms << " bedrooms: $" << output_price(0);

	session->Close();

	return 0;
}
