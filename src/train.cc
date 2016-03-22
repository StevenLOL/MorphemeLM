#include "train.h"

using namespace cnn;
using namespace cnn::expr;
using namespace cnn::mp;
using namespace std;
namespace po = boost::program_options;

class Learner : public ILearner<Sentence, SufficientStats> {
public:
  Learner(Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab, MorphLM& lm, Model& cnn_model) :
    word_vocab(word_vocab), root_vocab(root_vocab), affix_vocab(affix_vocab), char_vocab(char_vocab), lm(lm), cnn_model(cnn_model) {}
  ~Learner() {}
  SufficientStats LearnFromDatum(const Sentence& datum, bool learn) {
    ComputationGraph cg;
    lm.BuildGraph(datum, cg);
    cnn::real loss = as_scalar(cg.forward());
    if (learn) {
      cg.backward();
    }
    return SufficientStats(loss, datum.size(), 1);
  }

  void SaveModel() {
    if (!quiet) {
      Serialize(word_vocab, root_vocab, affix_vocab, char_vocab, lm, cnn_model);
    }
  }

  bool quiet;
private:
  Dict& word_vocab;
  Dict& root_vocab;
  Dict& affix_vocab;
  Dict& char_vocab;
  MorphLM& lm;
  Model& cnn_model;
};

// This function lets us elegantly handle the user pressing ctrl-c.
// We set a global flag, which causes the training loops to clean up
// and break. In particular, this allows models to be saved to disk
// before actually exiting the program.
bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    cerr << "Exiting..." << endl;
    exit(1);
  }
  else {
    cerr << "Ctrl-c pressed!" << endl;
    ctrlc_pressed = true;
    cnn::mp::stop_requested = true;
  }
}

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);
  cnn::Initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("train_text", po::value<string>()->required(), "Training text, morphologically analyzed")
  ("dev_text", po::value<string>()->required(), "Dev text, used for early stopping")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("cores,j", po::value<unsigned>()->default_value(1), "Number of CPU cores to use for training")
  ("dropout_rate", po::value<float>(), "Dropout rate (should be >= 0.0 and < 1)")
  ("report_frequency,r", po::value<unsigned>()->default_value(100), "Show the training loss of every r examples")
  ("dev_frequency,d", po::value<unsigned>()->default_value(10000), "Run the dev set every d examples. Save the model if the score is a new best")
  ("quiet,q", "Do not output model")
  ("model", po::value<string>(), "Reload this model and continue learning");

  AddTrainerOptions(desc);

  po::positional_options_description positional_options;
  positional_options.add("train_text", 1);
  positional_options.add("dev_text", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const unsigned num_cores = vm["cores"].as<unsigned>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const string train_text_filename = vm["train_text"].as<string>();
  const string dev_text_filename = vm["dev_text"].as<string>();

  Dict word_vocab, root_vocab, affix_vocab, char_vocab;
  Model cnn_model;
  MorphLM* lm = nullptr;
  Trainer* trainer = nullptr;

  if (vm.count("model")) {
    lm = new MorphLM();
    string model_filename = vm["model"].as<string>();
    Deserialize(model_filename, word_vocab, root_vocab, affix_vocab, char_vocab, *lm, cnn_model);
    assert (word_vocab.is_frozen());
    assert (root_vocab.is_frozen());
    assert (affix_vocab.is_frozen());
    assert (char_vocab.is_frozen());
  }

  // TODO: Do we need start symbols at all?
  word_vocab.Convert("UNK");
  word_vocab.Convert("<s>");
  word_vocab.Convert("</s>");
  root_vocab.Convert("UNK");
  root_vocab.Convert("<s>");
  root_vocab.Convert("</s>");
  char_vocab.Convert("UNK");
  char_vocab.Convert("<s>");
  char_vocab.Convert("</s>");
  char_vocab.Convert("</w>");
  affix_vocab.Convert("UNK");
  affix_vocab.Convert("</w>");

  vector<Sentence> train_text = ReadMorphText(train_text_filename, word_vocab, root_vocab, affix_vocab, char_vocab); 

  if (!vm.count("model")) {
    MorphLMConfig config;
    config.word_vocab_size = word_vocab.size();
    config.root_vocab_size = root_vocab.size();
    config.affix_vocab_size = affix_vocab.size();
    config.char_vocab_size = char_vocab.size();
    config.word_embedding_dim = 64;
    config.root_embedding_dim = 64;
    config.affix_embedding_dim = 64;
    config.char_embedding_dim = 16;
    config.model_chooser_hidden_dim = 16;
    config.affix_lstm_init_hidden_dim = 64;
    config.char_lstm_init_hidden_dim = 32;
    config.main_lstm_dim = 64;
    config.affix_lstm_dim = 64;
    config.char_lstm_dim = 32;
    lm = new MorphLM(cnn_model, config);

    word_vocab.Freeze();
    word_vocab.SetUnk("UNK");
    root_vocab.Freeze();
    root_vocab.SetUnk("UNK");
    affix_vocab.Freeze();
    affix_vocab.SetUnk("UNK");
    char_vocab.Freeze();
    char_vocab.SetUnk("UNK");
    cerr << "Dicts frozen" << endl;
  }

  vector<Sentence> dev_text = ReadMorphText(dev_text_filename, word_vocab, root_vocab, affix_vocab, char_vocab);

  if (vm.count("dropout_rate")) {
    lm->SetDropout(vm["dropout_rate"].as<float>());
  }

  cerr << "Vocabulary sizes: " << word_vocab.size() << " words, " << root_vocab.size() << " roots, " << affix_vocab.size() << " affixes, " << char_vocab.size() << " chars" << endl;
  cerr << "Total parameters: " << cnn_model.parameter_count() << endl;

  trainer = CreateTrainer(cnn_model, vm);
  Learner learner(word_vocab, root_vocab, affix_vocab, char_vocab, *lm, cnn_model);
  learner.quiet = vm.count("quiet") > 0;
  unsigned dev_frequency = vm["dev_frequency"].as<unsigned>();
  unsigned report_frequency = vm["report_frequency"].as<unsigned>();
  if (num_cores > 1) {
    RunMultiProcess<Sentence>(num_cores, &learner, trainer, train_text, dev_text, num_iterations, dev_frequency, report_frequency);
  }
  else {
    RunSingleProcess<Sentence>(&learner, trainer, train_text, dev_text, num_iterations, dev_frequency, report_frequency);
  }

  return 0;
}
