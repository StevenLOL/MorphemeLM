#include "morphlm.h"
#define SAFE_DELETE(p) if ((p) != nullptr) { delete (p); (p) = nullptr; }

const unsigned lstm_layer_count = 2;

MorphLM::MorphLM() {}

MorphLM::~MorphLM() {
  SAFE_DELETE(word_softmax);
  SAFE_DELETE(root_softmax);
  SAFE_DELETE(affix_softmax);
  SAFE_DELETE(char_softmax);
}

MorphLM::MorphLM(Model& model, const MorphLMConfig& config) {
  this->config = config;

  input_word_embeddings = model.add_lookup_parameters(config.word_vocab_size, {config.word_embedding_dim});
  input_root_embeddings = model.add_lookup_parameters(config.root_vocab_size, {lstm_layer_count * config.affix_lstm_dim});
  input_affix_embeddings = model.add_lookup_parameters(config.affix_vocab_size, {config.affix_embedding_dim});
  input_char_embeddings = model.add_lookup_parameters(config.char_vocab_size, {config.char_embedding_dim});

  input_char_lstm_init = model.add_parameters({lstm_layer_count * config.char_lstm_dim});

  input_affix_lstm = LSTMBuilder(lstm_layer_count, config.affix_embedding_dim, config.affix_lstm_dim, &model);
  input_char_lstm = LSTMBuilder(lstm_layer_count, config.char_embedding_dim, config.char_lstm_dim, &model);

  unsigned total_input_dim = config.word_embedding_dim + config.affix_lstm_dim + config.char_lstm_dim;
  main_lstm_init = model.add_parameters({lstm_layer_count * config.main_lstm_dim});
  main_lstm = LSTMBuilder(lstm_layer_count, total_input_dim, config.main_lstm_dim, &model);
  model_chooser = MLP(model, config.main_lstm_dim, config.model_chooser_hidden_dim, 4);

  word_softmax = new StandardSoftmaxBuilder(config.main_lstm_dim, config.word_vocab_size, &model);
  root_softmax = new StandardSoftmaxBuilder(config.main_lstm_dim, config.root_vocab_size, &model);
  affix_softmax = new StandardSoftmaxBuilder(config.affix_lstm_dim, config.affix_vocab_size, &model);
  char_softmax = new StandardSoftmaxBuilder(config.char_lstm_dim, config.char_vocab_size, &model);

  output_root_embeddings = model.add_lookup_parameters(config.root_vocab_size, {config.root_embedding_dim});
  output_affix_embeddings = model.add_lookup_parameters(config.affix_vocab_size, {config.affix_embedding_dim});
  output_char_embeddings = model.add_lookup_parameters(config.char_vocab_size, {config.char_embedding_dim});

  output_char_lstm_init = MLP(model, config.main_lstm_dim, config.char_lstm_init_hidden_dim, lstm_layer_count * config.char_lstm_dim);
  output_affix_lstm_init = MLP(model, config.main_lstm_dim + config.root_embedding_dim, config.affix_lstm_init_hidden_dim, lstm_layer_count * config.affix_lstm_dim);

  output_affix_lstm = LSTMBuilder(lstm_layer_count, config.affix_embedding_dim + config.main_lstm_dim, config.affix_lstm_dim, &model);
  output_char_lstm = LSTMBuilder(lstm_layer_count, config.char_embedding_dim + config.main_lstm_dim, config.char_lstm_dim, &model);
}

void MorphLM::NewGraph(ComputationGraph& cg) {
  Expression input_char_lstm_init_expr = parameter(cg, input_char_lstm_init);
  input_char_lstm_init_v = MakeLSTMInitialState(input_char_lstm_init_expr, config.char_lstm_dim, lstm_layer_count); 

  input_affix_lstm.new_graph(cg);
  input_char_lstm.new_graph(cg);

  Expression main_lstm_init_expr = parameter(cg, main_lstm_init);
  main_lstm_init_v = MakeLSTMInitialState(main_lstm_init_expr, config.main_lstm_dim, lstm_layer_count); 

  main_lstm.new_graph(cg);
  model_chooser.NewGraph(cg);

  word_softmax->new_graph(cg);
  root_softmax->new_graph(cg);
  affix_softmax->new_graph(cg);
  char_softmax->new_graph(cg);

  output_affix_lstm_init.NewGraph(cg);
  output_char_lstm_init.NewGraph(cg);

  output_affix_lstm.new_graph(cg);
  output_char_lstm.new_graph(cg);
}

Expression MorphLM::BuildGraph(const Sentence& sentence, ComputationGraph& cg) {
  assert (sentence.size() > 0);
  NewGraph(cg);

  vector<Expression> inputs;
  for (unsigned i = 0; i < sentence.size(); ++i) {
    Expression word_embedding = EmbedWord(sentence.words[i], cg);
    Expression analysis_embedding = EmbedAnalyses(sentence.analyses[i], sentence.analysis_probs[i], cg);
    Expression char_embedding = EmbedCharacterSequence(sentence.chars[i], cg);

    Expression input_embedding = concatenate({word_embedding, analysis_embedding, char_embedding});
    inputs.push_back(input_embedding);
  }

  vector<Expression> losses;
  main_lstm.start_new_sequence();
  for (unsigned i = 0; i < inputs.size(); ++i) {
    Expression context = main_lstm.add_input(inputs[i]);
    Expression mode_log_probs = log_softmax(model_chooser.Feed(context));
    if (i == inputs.size() - 1) {
      assert (sentence.words[i] == 2); // </s>
      Expression loss = -pick(mode_log_probs, (unsigned)0);
      losses.push_back(loss);
      break;
    }

    Expression word_loss = ComputeWordLoss(context, sentence.words[i], cg);
    Expression morpheme_loss = ComputeMorphemeLoss(context, sentence.analyses[i], sentence.analysis_probs[i], cg);
    Expression char_loss = ComputeCharLoss(context, sentence.chars[i], cg);

    // Have -log p(w | c, m) for each of the three values of m
    // Want total_loss = -log p(w | c).
    // p(w | c) = \sum_M p(w | c, m) p(m)
    // so log p(w | c) = logsumexp_M log p(w | c, m) + log p(m)
    // so total_loss = -logsumexp_M -mode_losses + mode_log_probs
    // = -logsumexp(mode_log_probs - mode_losses);
    word_loss = pick(mode_log_probs, 1) - word_loss;
    morpheme_loss = pick(mode_log_probs, 2) - morpheme_loss;
    char_loss = pick(mode_log_probs, 3) - char_loss;

    Expression total_loss = -logsumexp({word_loss, morpheme_loss, char_loss});

    losses.push_back(total_loss);
  }

  assert (losses.size() == sentence.size());

  return sum(losses);
}

void MorphLM::SetDropout(float r) {
}

Expression MorphLM::EmbedWord(const WordId word, ComputationGraph& cg) {
  return lookup(cg, input_word_embeddings, word);
}

Expression MorphLM::EmbedAnalysis(const Analysis& analysis, ComputationGraph& cg) {

  Expression root_embedding = lookup(cg, input_root_embeddings, analysis.root);
  vector<Expression> hinit = MakeLSTMInitialState(root_embedding, config.affix_lstm_dim, lstm_layer_count);
  input_affix_lstm.start_new_sequence(hinit);

  for (WordId affix : analysis.affixes) {
    Expression affix_embedding = lookup(cg, input_affix_embeddings, affix);
    input_affix_lstm.add_input(affix_embedding);
  }

  return input_affix_lstm.back();
}

Expression MorphLM::EmbedAnalyses(const vector<Analysis>& analyses, const vector<float>& probs, ComputationGraph& cg) {
  assert (analyses.size() > 0);
  vector<Expression> analysis_embeddings(analyses.size());
  for (unsigned i = 0; i < analyses.size(); ++i) {
    analysis_embeddings[i] = EmbedAnalysis(analyses[i], cg);
  }

  // Ghetto max pooling
  Expression final_embedding = analysis_embeddings[0];
  for (unsigned i = 1; i < analyses.size(); ++i) {
    final_embedding = max(final_embedding, analysis_embeddings[i]);
  }

  return final_embedding;
}

Expression MorphLM::EmbedCharacterSequence(const vector<WordId>& chars, ComputationGraph& cg) {
  input_char_lstm.start_new_sequence(input_char_lstm_init_v);
  for (WordId c : chars) {
    Expression char_embedding = lookup(cg, input_char_embeddings, c);
    input_char_lstm.add_input(char_embedding);
  }
  return input_char_lstm.back();
}

Expression MorphLM::ComputeWordLoss(Expression context, WordId ref, ComputationGraph& cg) {
  return word_softmax->neg_log_softmax(context, ref);
}

Expression MorphLM::ComputeAnalysisLoss(Expression context, const Analysis& ref, ComputationGraph& cg) {
  Expression root_loss = root_softmax->neg_log_softmax(context, ref.root);

  Expression root_embedding = lookup(cg, output_root_embeddings, ref.root);
  Expression c = output_affix_lstm_init.Feed(concatenate({root_embedding, context}));
  vector<Expression> hinit = MakeLSTMInitialState(c, config.affix_lstm_dim, lstm_layer_count);

  output_affix_lstm.start_new_sequence(hinit);
  vector<Expression> losses;
  losses.push_back(root_loss);
  for (WordId affix : ref.affixes) {
    Expression h = output_affix_lstm.back();
    Expression loss = affix_softmax->neg_log_softmax(h, affix);
    losses.push_back(loss);

    Expression affix_embedding =lookup(cg, output_affix_embeddings, affix);
    Expression input = concatenate({affix_embedding, context});
    output_affix_lstm.add_input(input);
  }
  return sum(losses);
}

Expression MorphLM::ComputeMorphemeLoss(Expression context, const vector<Analysis>& refs, const vector<float>& probs, ComputationGraph& cg) {
  vector<Expression> losses(refs.size());
  for (unsigned i = 0; i < refs.size(); ++i) {
    losses[i] = ComputeAnalysisLoss(context, refs[i], cg);
  }
  return logsumexp(losses);
}

Expression MorphLM::ComputeCharLoss(Expression context, const vector<WordId>& ref, ComputationGraph& cg) {
  Expression c = output_char_lstm_init.Feed(context);
  vector<Expression> hinit = MakeLSTMInitialState(c, config.char_lstm_dim, lstm_layer_count);
  output_char_lstm.start_new_sequence(hinit);

  vector<Expression> losses;
  for (WordId w : ref) {
    Expression h = output_char_lstm.back();
    Expression char_loss = char_softmax->neg_log_softmax(h, w);
    losses.push_back(char_loss);

    Expression char_embedding = lookup(cg, output_char_embeddings, w);
    Expression input = concatenate({char_embedding, context});
    output_char_lstm.add_input(input);
  }

  return sum(losses);
}

Sentence MorphLM::Sample(unsigned max_length) {
  Sentence sentence;
  return sentence;
}

vector<Expression> MakeLSTMInitialState(Expression c, unsigned lstm_dim, unsigned lstm_layer_count) {
  vector<Expression> hinit(lstm_layer_count * 2);
  for (unsigned i = 0; i < lstm_layer_count; ++i) {
    hinit[i] = pickrange(c, i * lstm_dim, (i + 1) * lstm_dim);
    hinit[i + lstm_layer_count] = tanh(hinit[i]);
  }
  return hinit;
}
