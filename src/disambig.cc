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
  lm.SetDropout(0.0f);
  cerr << "Loading model from " << model_filename << "...";
  Deserialize(model_filename, word_vocab, root_vocab, affix_vocab, char_vocab, lm, dynet_model);
  cerr << " Done!" << endl;

  unsigned sentence_number = 0;
  Sentence input;
  while(ReadMorphSentence(cin, word_vocab, root_vocab, affix_vocab, char_vocab, input)) {
    ComputationGraph cg;
    lm.NewGraph(cg);
    lm.main_lstm.start_new_sequence(lm.main_lstm_init_v);
    for (unsigned i = 0; i < input.analyses.size(); ++i) {
      vector<float> losses;
      Expression context = lm.main_lstm.back();
      Expression embedding = lm.EmbedInput(input, i, cg);
      for (unsigned j = 0; j < input.analyses[i].size(); ++j) {
        const Analysis& analysis = input.analyses[i][j];
        Expression loss = lm.ComputeAnalysisLoss(context, analysis, cg);
        losses.push_back(-as_scalar(loss.value()));
      }
      assert (losses.size() == input.analyses[i].size());
      cout << sentence_number << " ||| " << i << " |||";
      float sum = logsumexp(losses);
      for (unsigned j = 0; j < input.analyses[i].size(); ++j) {
        cout << " " << expf(losses[j] - sum);
      }
      cout << endl;
      lm.main_lstm.add_input(embedding);
    }
    cout << endl;
    cout.flush();
    sentence_number++;
  }

  return 0;
}
