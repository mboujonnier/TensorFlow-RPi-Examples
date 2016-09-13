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
	string ograph = "/home/tas/ML/tensorflow/tensorflow/contrib/pi_examples/linear_regression/data/output_graph.pb";

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

	/*
	python equivalent code :
	test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
	test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
	print("Testing... (Mean square loss Comparison)")
	cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0])
	testing_cost = sess.run(cost, feed_dict={X: test_X, Y: test_Y})
	print("Testing cost=", testing_cost)
	*/

	// Create the test data to check accuracy
	LOG(INFO) << "Generate Test data (X and Y)";
	float test_X[8] = { 6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1 };
	float test_Y[8] = { 1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03 };

	tensorflow::Tensor inputs_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 1, 8, 1}));
	tensorflow::Tensor size_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 1, 1, 1}));
	tensorflow::Tensor results_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 1, 8, 1}));

	auto inputs_tensor_mapped = inputs_tensor.tensor<float, 4>();
	auto size_tensor_mapped = size_tensor.tensor<float, 4>();
	auto results_tensor_mapped = results_tensor.tensor<float, 4>();

	LOG(INFO) << "Creating 2 tensors from C++ array : ";
	size_tensor_mapped(0)=8;
	for(int i=0; i<8; i++)
	{
		inputs_tensor_mapped(0, 0, i, 0) = test_X[i];
		results_tensor_mapped(0, 0, i, 0) = test_Y[i];
	}

	// check tensors
	//auto test = inputs_tensor.flat<float>();
	//LOG(INFO) << "inputs: " << test(0);
	//LOG(INFO) << "inputs: " << test(1);
	//LOG(INFO) << "inputs: " << test(2);

	LOG(INFO) << "Run the model";
	// virtual Status tensorflow::Session::Run(const std::vector< std::pair< string, Tensor > > &inputs, const std::vector< string > &output_tensor_names,	const std::vector< string > &target_node_names,	std::vector< Tensor > *outputs)=0
	Status run_status = session->Run({{"input_node/X", inputs_tensor}, {"input_node/Y", results_tensor}, {"input_node/N", size_tensor}}, {{"output_weight", "output_bias", "output_cost", "output_training_cost"}}, {}, &outputs);
	if (!run_status.ok())
	{
		LOG(ERROR) << "Running model failed: " << run_status;
		return -1;
	}

	LOG(INFO) << "Checking  model accuracy with test data";
	auto output_weight = 	outputs[0].flat<float>();
	auto output_bias = 		outputs[1].flat<float>();
	auto output_cost = 		outputs[2].flat<float>();
	auto output_training_cost = 	outputs[3].flat<float>();

	LOG(INFO) << "w: " << output_weight(0);
	LOG(INFO) << "b: " << output_bias(0);
	LOG(INFO) << "testing cost: " << output_cost(0);
	LOG(INFO) << "training cost: " << output_training_cost(0);

	if (output_cost(0) > 0.5)
	{
		LOG(INFO) << "Linear model is not accurate anymore, testing cost " << output_cost(0);
	}
	else
	{
		LOG(INFO) << "Absolute mean square loss difference:" << (output_training_cost(0) - output_cost(0));
	}

	session->Close();

	return 0;
}
