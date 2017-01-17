#include <boost/algorithm/string/join.hpp>
#include "morphlm.h"
#define SAFE_DELETE(p) if ((p) != nullptr) { delete (p); (p) = nullptr; }

const unsigned lstm_layer_count = 2;

MorphLM::MorphLM() :
    word_softmax(nullptr), root_softmax(nullptr), affix_softmax(nullptr), char_softmax(nullptr) {}

MorphLM::~MorphLM() {
  SAFE_DELETE(word_softmax);
  SAFE_DELETE(root_softmax);
  SAFE_DELETE(affix_softmax);
  SAFE_DELETE(char_softmax);
}

MorphLM::MorphLM(Model& model, const MorphLMConfig& config) :
    word_softmax(nullptr), root_softmax(nullptr), affix_softmax(nullptr), char_softmax(nullptr) {
  this->config = config;

  if (config.use_words) {
    input_word_embeddings = model.add_lookup_parameters(config.word_vocab_size, {config.word_embedding_dim});
  }
  if (config.use_morphology) {
    input_root_embeddings = model.add_lookup_parameters(config.root_vocab_size, {lstm_layer_count * config.affix_lstm_dim});
    input_affix_embeddings = model.add_lookup_parameters(config.affix_vocab_size, {config.affix_embedding_dim});
  }
  input_char_embeddings = model.add_lookup_parameters(config.char_vocab_size, {config.char_embedding_dim});

  input_char_lstm_init = model.add_parameters({lstm_layer_count * config.char_lstm_dim});

  if (config.use_morphology) {
    input_affix_lstm = LSTMBuilder(lstm_layer_count, config.affix_embedding_dim, config.affix_lstm_dim, model);
  }
  input_char_lstm = LSTMBuilder(lstm_layer_count, config.char_embedding_dim, config.char_lstm_dim, model);

  unsigned total_input_dim = config.char_lstm_dim;
  unsigned output_mode_count = 2; // char-level or EOS
  if (config.use_morphology) {
    total_input_dim += config.affix_lstm_dim;
    output_mode_count++;
  }
  if (config.use_words) {
    total_input_dim += config.word_embedding_dim;
    output_mode_count++;
  }

  unsigned context_dim = config.bidirectional ? 2 * config.main_lstm_dim : config.main_lstm_dim;
  main_lstm_fwd_init = model.add_parameters({lstm_layer_count * config.main_lstm_dim});
  main_lstm_fwd = LSTMBuilder(lstm_layer_count, total_input_dim, config.main_lstm_dim, model);
  if (config.bidirectional) {
    main_lstm_rev_init = model.add_parameters({lstm_layer_count * config.main_lstm_dim});
    main_lstm_rev = LSTMBuilder(lstm_layer_count, total_input_dim, config.main_lstm_dim, model);
  }
  model_chooser = MLP(model, context_dim, config.model_chooser_hidden_dim, output_mode_count);

  if (config.use_words) {
    word_softmax = new StandardSoftmaxBuilder(context_dim, config.word_vocab_size, model);
  }
  if (config.use_morphology) {
    root_softmax = new StandardSoftmaxBuilder(context_dim, config.root_vocab_size, model);
    affix_softmax = new StandardSoftmaxBuilder(config.affix_lstm_dim, config.affix_vocab_size, model);
  }
  char_softmax = new StandardSoftmaxBuilder(config.char_lstm_dim, config.char_vocab_size, model);

  if (config.use_morphology) {
    output_root_embeddings = model.add_lookup_parameters(config.root_vocab_size, {config.root_embedding_dim});
    output_affix_embeddings = model.add_lookup_parameters(config.affix_vocab_size, {config.affix_embedding_dim});
  }
  output_char_embeddings = model.add_lookup_parameters(config.char_vocab_size, {config.char_embedding_dim});

  output_char_lstm_init = MLP(model, context_dim, config.char_lstm_init_hidden_dim, lstm_layer_count * config.char_lstm_dim);
  if (config.use_morphology) {
    output_affix_lstm_init = MLP(model, context_dim + config.root_embedding_dim, config.affix_lstm_init_hidden_dim, lstm_layer_count * config.affix_lstm_dim);
  }

  if (config.use_morphology) {
    output_affix_lstm = LSTMBuilder(lstm_layer_count, config.affix_embedding_dim + context_dim, config.affix_lstm_dim, model);
  }
  output_char_lstm = LSTMBuilder(lstm_layer_count, config.char_embedding_dim + context_dim, config.char_lstm_dim, model);
}

void MorphLM::NewGraph(ComputationGraph& cg) {
  Expression input_char_lstm_init_expr = parameter(cg, input_char_lstm_init);
  input_char_lstm_init_v = MakeLSTMInitialState(input_char_lstm_init_expr, config.char_lstm_dim, lstm_layer_count);

  if (config.use_morphology) {
    input_affix_lstm.new_graph(cg);
  }
  input_char_lstm.new_graph(cg);

  Expression main_lstm_fwd_init_expr = parameter(cg, main_lstm_fwd_init);
  main_lstm_fwd_init_v = MakeLSTMInitialState(main_lstm_fwd_init_expr, config.main_lstm_dim, lstm_layer_count);
  main_lstm_fwd.new_graph(cg);

  if (config.bidirectional) {
    Expression main_lstm_rev_init_expr = parameter(cg, main_lstm_rev_init);
    main_lstm_rev_init_v = MakeLSTMInitialState(main_lstm_rev_init_expr, config.main_lstm_dim, lstm_layer_count);
    main_lstm_rev.new_graph(cg);
  }

  model_chooser.NewGraph(cg);

  if (config.use_words) {
    word_softmax->new_graph(cg);
  }
  if (config.use_morphology) {
    root_softmax->new_graph(cg);
    affix_softmax->new_graph(cg);
  }
  char_softmax->new_graph(cg);

  if (config.use_morphology) {
    output_affix_lstm_init.NewGraph(cg);
  }
  output_char_lstm_init.NewGraph(cg);

  if (config.use_morphology) {
    output_affix_lstm.new_graph(cg);
  }
  output_char_lstm.new_graph(cg);
}

Expression MorphLM::EmbedInput(const Sentence& sentence, unsigned i, ComputationGraph& cg) {
  vector<Expression> mode_embeddings;
  Expression char_embedding = EmbedCharacterSequence(sentence.chars[i], cg);
  mode_embeddings.push_back(char_embedding);
  if (config.use_morphology) {
    Expression analysis_embedding = EmbedAnalyses(sentence.analyses[i], sentence.analysis_probs[i], cg);
    mode_embeddings.push_back(analysis_embedding);
  }
  if (config.use_words) {
    Expression word_embedding = EmbedWord(sentence.words[i], cg);
    mode_embeddings.push_back(word_embedding);
  }

  Expression input_embedding = concatenate(mode_embeddings);
  //Expression input_embedding = sum(mode_embeddings);
  return input_embedding;
}

vector<Expression> MorphLM::ShowModeProbs(const Sentence& sentence, ComputationGraph& cg) {
  assert (sentence.size() > 0);
  NewGraph(cg);

  vector<Expression> inputs;
  for (unsigned i = 0; i < sentence.size(); ++i) {
    Expression input_embedding = EmbedInput(sentence, i, cg);
    inputs.push_back(input_embedding);
  }

  vector<Expression> mode_exprs;
  vector<Expression> contexts = GetContexts(inputs, cg);
  for (unsigned i = 0; i < inputs.size(); ++i) {
    Expression context = contexts[i];
    Expression mode_log_probs = log_softmax(model_chooser.Feed(context));
    mode_exprs.push_back(mode_log_probs);
  }

  assert (mode_exprs.size() == sentence.size());

  return mode_exprs;
}

vector<Expression> MorphLM::GetContexts(const vector<Expression>& inputs, ComputationGraph& cg) {
  vector<Expression> context_vectors(inputs.size());
  main_lstm_fwd.start_new_sequence(main_lstm_fwd_init_v);
  for (unsigned i = 0; i < inputs.size(); ++i) {
    Expression context = main_lstm_fwd.back();
    context_vectors[i] = context;
    main_lstm_fwd.add_input(inputs[i]);
  }

  if (config.bidirectional) {
    main_lstm_rev.start_new_sequence(main_lstm_rev_init_v);
    for (unsigned i = 0; i < inputs.size(); ++i) {
      unsigned j = inputs.size() - 1 - i;
      Expression context = main_lstm_rev.back();
      context_vectors[j] = concatenate({context_vectors[j], context});
      main_lstm_rev.add_input(inputs[j]);
    }
  }

  return context_vectors;
}

Expression MorphLM::BuildGraph(const Sentence& sentence, ComputationGraph& cg) {
  assert (sentence.size() > 0);
  NewGraph(cg);

  vector<Expression> inputs;
  for (unsigned i = 0; i < sentence.size(); ++i) {
    Expression input_embedding = EmbedInput(sentence, i, cg);
    inputs.push_back(input_embedding);
  }

  vector<Expression> losses;
  vector<Expression> context_vectors = GetContexts(inputs, cg);;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    Expression& context = context_vectors[i];
    Expression mode_log_probs = log_softmax(model_chooser.Feed(context));
    if (i == inputs.size() - 1) {
      assert (sentence.words[i] == 2); // </s>
      Expression loss = -pick(mode_log_probs, (unsigned)0);
      losses.push_back(loss);
      break;
    }

    // Have -log p(w | c, m) for each of the three values of m
    // Want total_loss = -log p(w | c).
    // p(w | c) = \sum_M p(w | c, m) p(m)
    // so log p(w | c) = logsumexp_M log p(w | c, m) + log p(m)
    // so total_loss = -logsumexp_M -mode_losses + mode_log_probs
    // = -logsumexp(mode_log_probs - mode_losses);

    vector<Expression> mode_losses;
    unsigned mode_index = 1;

    Expression char_loss = ComputeCharLoss(context, sentence.chars[i], cg);
    char_loss = pick(mode_log_probs, mode_index++) - char_loss;
    mode_losses.push_back(char_loss);

    if (config.use_morphology) {
      if (sentence.analyses[i].size() > 0 and sentence.analyses[i][0].root != 0) {
        Expression morpheme_loss = ComputeMorphemeLoss(context, sentence.analyses[i], sentence.analysis_probs[i], cg);
        morpheme_loss = pick(mode_log_probs, mode_index++) - morpheme_loss;
        mode_losses.push_back(morpheme_loss);
      }
    }

    if (config.use_words) {
      if (sentence.words[i] != 0) {
        Expression word_loss = ComputeWordLoss(context, sentence.words[i], cg);
        word_loss = pick(mode_log_probs, mode_index++) - word_loss;
        mode_losses.push_back(word_loss);
      }
    }

    Expression total_loss = -logsumexp(mode_losses);

    losses.push_back(total_loss);
  }

  assert (losses.size() == sentence.size());

  return sum(losses);
}

void MorphLM::SetDropout(float r) {
  main_lstm_fwd.set_dropout(r);
  if (config.bidirectional) {
    main_lstm_rev.set_dropout(r);
  }
  if (config.use_morphology) {
    input_affix_lstm.set_dropout(r);
    output_affix_lstm.set_dropout(r);
  }
  input_char_lstm.set_dropout(r);
  output_char_lstm.set_dropout(r);
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

vector<WordId> MorphLM::SampleCharSequence(Expression context, unsigned max_length, ComputationGraph& cg) {
  Expression c = output_char_lstm_init.Feed(context);
  vector<Expression> hinit = MakeLSTMInitialState(c, config.char_lstm_dim, lstm_layer_count);
  output_char_lstm.start_new_sequence(hinit);

  vector<WordId> seq;
  while (seq.size() < max_length) {
    Expression h = output_char_lstm.back();
    WordId w = char_softmax->sample(h);

    if (w == 3) {
      seq.push_back(w);
      break;
    }
    else {
      seq.push_back(w);
      Expression char_embedding = lookup(cg, output_char_embeddings, w);
      Expression input = concatenate({char_embedding, context});
      output_char_lstm.add_input(input);
    }
  }
  return seq;
}

Analysis MorphLM::SampleMorphAnalysis(Expression context, unsigned max_length, ComputationGraph& cg) {
  WordId root = root_softmax->sample(context);
  Expression root_embedding = lookup(cg, output_root_embeddings, root);

  Expression c = output_affix_lstm_init.Feed(concatenate({root_embedding, context}));
  vector<Expression> hinit = MakeLSTMInitialState(c, config.affix_lstm_dim, lstm_layer_count);
  output_affix_lstm.start_new_sequence(hinit);

  vector<WordId> affixes;
  while (affixes.size() < max_length) {
    Expression h = output_affix_lstm.back();
    WordId affix = affix_softmax->sample(h);

    if (affix == 3) {
      break;
    }
    else {
      affixes.push_back(affix);
      Expression affix_embedding = lookup(cg, output_affix_embeddings, affix);
      Expression input = concatenate({affix_embedding, context});
      output_affix_lstm.add_input(input);
    }
  }
  return Analysis {root, affixes};
}

Sentence MorphLM::Sample(unsigned max_length, ComputationGraph& cg, WordFillerOuter* wfo) {
  assert (!config.bidirectional && "Sampling not supported in bidirectional mode!");
  random_device rd;
  mt19937 rng(rd());
  Sentence sentence;

  NewGraph(cg);
  main_lstm_fwd.start_new_sequence(main_lstm_fwd_init_v);

  for (unsigned word_index = 0; word_index <= max_length; ++word_index) {
    Expression context = main_lstm_fwd.back();
    Expression mode_log_probs = log_softmax(model_chooser.Feed(context));
    vector<float> mode_log_prob_vals = as_vector(mode_log_probs.value());
    assert (!config.use_morphology); // HACK: For now this doesn't work with morphology-enabled models.

    unsigned mode = sample_multinomial(mode_log_prob_vals, rng);
    if (mode == 0) {
      // We sampled </s>
      break;
    }
    else if (mode == 1) {
      // Generate a character sequence
      vector<WordId> char_seq = SampleCharSequence(context, 100, cg);
      sentence.words.push_back(0);
      sentence.analyses.push_back(vector<Analysis>());
      sentence.analysis_probs.push_back(vector<float>());
      sentence.chars.push_back(char_seq);
      //wfo->FillFromChars(sentence);
    }
    else if (mode == 2 && config.use_morphology) {
      // Generate a morpheme sequence
      Analysis analysis = SampleMorphAnalysis(context, 100, cg);
      sentence.words.push_back(0);
      sentence.analyses.push_back(vector<Analysis>());
      sentence.analysis_probs.push_back(vector<float>());
      sentence.chars.push_back(vector<WordId>());

      sentence.analysis_probs.back().push_back(1.0f);
      sentence.analyses.back().push_back(analysis);
      wfo->FillFromMorph(sentence);
    }
    else {
      // Generate a word directly
      WordId word_id = word_softmax->sample(context);
      sentence.words.push_back(word_id);
      sentence.analyses.push_back(vector<Analysis>());
      sentence.analysis_probs.push_back(vector<float>());
      sentence.chars.push_back(vector<WordId>());
      //wfo->FillFromWord(sentence);
    }
    Expression input_embedding = EmbedInput(sentence, word_index, cg);
    main_lstm_fwd.add_input(input_embedding);
  }
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

void WordFillerOuter::FillFromChars(Sentence& sentence) {
  assert (char_vocab != nullptr);
  vector<WordId>& char_ids = sentence.chars.back();
  vector<string> chars(char_ids.size());
  for (unsigned i = 0; i < char_ids.size(); ++i) {
    chars[i] = char_vocab->convert(char_ids[i]);
  }
  string word = boost::algorithm::join(chars, "");

  if (root_vocab != nullptr) {
    assert (affix_vocab != nullptr);
    assert (false);
  }

  if (word_vocab != nullptr) {
    sentence.words.back() = word_vocab->convert(word);
  }  
}

void WordFillerOuter::FillFromMorph(Sentence& sentence) {
  assert (root_vocab != nullptr);
  assert (affix_vocab != nullptr);
  assert (false);
}

void WordFillerOuter::FillFromWord(Sentence& sentence) {
  assert (word_vocab != nullptr);
  string word = word_vocab->convert(sentence.words.back());

  if (root_vocab != nullptr) {
    assert (affix_vocab != nullptr);
    assert (false);
  }

  vector<WordId>& chars = sentence.chars.back();
  assert (chars.size() == 0);
  if (char_vocab != nullptr) {
    unsigned i = 0;
    while (i < word.length()) {
      unsigned len = UTF8Len(word[i]);
      string c = word.substr(i, len);
      chars.push_back(char_vocab->convert(c));
      i += len;
    }
    chars.push_back(char_vocab->convert("</w>"));
  }
}
