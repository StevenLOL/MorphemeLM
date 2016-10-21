#include "dynet/dynet.h"
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>

#include "io.h"
#include "utils.h"

using namespace dynet;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "model files, as output by train")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("model", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const string model_filename = vm["model"].as<string>();

  Model dynet_model;
  Dict word_vocab, root_vocab, affix_vocab, char_vocab;
  MorphLM lm;
  cerr << "Loading model from " << model_filename << "...";
  Deserialize(model_filename, word_vocab, root_vocab, affix_vocab, char_vocab, lm, dynet_model);
  cerr << " Done!" << endl;

  unsigned sentence_number = 0;
  Sentence input;
  while(ReadMorphSentence(cin, word_vocab, root_vocab, affix_vocab, char_vocab, input)) {
    ComputationGraph cg;
    vector<Expression> mode_log_probs = lm.ShowModeProbs(input, cg);
    for (Expression e : mode_log_probs) {
      vector<float> v = as_vector(e.value());
      for (unsigned i = 0; i < v.size(); ++i) {
        cout << ((i != 0) ? " " : "") << v[i];
      }
      cout << endl;
    }
    cout << endl;
    cout.flush();

    sentence_number++;
  }

  return 0;
}
