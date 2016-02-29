#include <fstream>
#include "io.h"

void Serialize(const Dict& dict, const MorphLM& lm, Model& cnn_model) {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {}
  fseek(stdout, 0, SEEK_SET);

  boost::archive::binary_oarchive oa(cout);
  oa & cnn_model;
  oa & dict;
  oa & lm;
}

void Deserialize(const string& filename, Dict& dict, MorphLM& lm, Model& cnn_model) {
  ifstream f(filename);
  boost::archive::binary_iarchive ia(f);
  ia & cnn_model;
  ia & dict;
  ia & lm;
  f.close();

  assert (dict.is_frozen());
}

