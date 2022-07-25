# DistilBERT changes

DistilBERT scripts were taken from [BERT scripts](../bert).

## Changed files

Files used:
* modeling.py
* optimization.py
* run_squad.py
* distilbert_squad_main.py
* download
  * download_pretrained_model.py

### modeling.py

* Changed scope to align with the pre-trained Huggingface model:
```python
with tf.compat.v1.variable_scope(scope, default_name="bert", custom_getter=get_custom_getter(compute_type)):
  with tf.compat.v1.variable_scope("embeddings"):
    # For good convergence with mixed precision training,
      # it is important that the embedding codes remain fp32.
    # Perform embedding lookup on the word ids.
    (self.embedding_output, self.embedding_table) = embedding_lookup(
        input_ids=input_ids,
        vocab_size=config.vocab_size,
        embedding_size=config.hidden_size,
        initializer_range=config.initializer_range,
        word_embedding_name="word_embeddings",
        use_one_hot_embeddings=use_one_hot_embeddings)

    # Add positional embeddings and token type embeddings, then layer
    # normalize and perform dropout.
    self.embedding_output = embedding_postprocessor(
        input_tensor=self.embedding_output,
        use_token_type=True,
        token_type_ids=token_type_ids,
        token_type_vocab_size=config.type_vocab_size,
        token_type_embedding_name="token_type_embeddings",
        use_position_embeddings=True,
        position_embedding_name="position_embeddings",
        initializer_range=config.initializer_range,
        max_position_embeddings=config.max_position_embeddings,
        dropout_prob=config.hidden_dropout_prob,
        use_one_hot_embeddings=use_one_hot_embeddings)

  with tf.compat.v1.variable_scope("encoder"):
```
to
```python
with tf.compat.v1.variable_scope(scope, default_name="model/distilbert", custom_getter=get_custom_getter(compute_type)):
  with tf.compat.v1.variable_scope("embeddings"):
    # For good convergence with mixed precision training,
      # it is important that the embedding codes remain fp32.
    # Perform embedding lookup on the word ids.
    (self.embedding_output, self.embedding_table) = embedding_lookup(
        input_ids=input_ids,
        vocab_size=config.vocab_size,
        embedding_size=config.hidden_size,
        initializer_range=config.initializer_range,
        word_embedding_name="weight",
        use_one_hot_embeddings=use_one_hot_embeddings)

    # Add positional embeddings and token type embeddings, then layer
    # normalize and perform dropout.
    self.embedding_output = embedding_postprocessor(
        input_tensor=self.embedding_output,
        use_token_type=True,
        token_type_ids=token_type_ids,
        token_type_vocab_size=config.type_vocab_size,
        token_type_embedding_name="token_type_embeddings",
        use_position_embeddings=True,
        position_embedding_name="embeddings",
        initializer_range=config.initializer_range,
        max_position_embeddings=config.max_position_embeddings,
        dropout_prob=config.hidden_dropout_prob,
        use_one_hot_embeddings=use_one_hot_embeddings)

  with tf.compat.v1.variable_scope("transformer"):
```

* Changed word_embedding_name in embedding_lookup function:
```python
def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
```
to
```python
def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="weight",
                     use_one_hot_embeddings=False):
```
to align with the pre-trained Huggingface model.

* Changed position_embedding_name in embedding_postprocessing function:
```python
def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1,
                            use_one_hot_embeddings=False):
```
to
```python
def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1,
                            use_one_hot_embeddings=False):
```
to align with the pre-trained Huggingface model.

* Changed query/key/value name in attention_layer function:
```python
query_layer = tf.compat.v1.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

# `key_layer` = [B*T, N*H]
key_layer = tf.compat.v1.layers.dense(
    to_tensor_2d,
    num_attention_heads * size_per_head,
    activation=key_act,
    name="key",
    kernel_initializer=create_initializer(initializer_range))

# `value_layer` = [B*T, N*H]
value_layer = tf.compat.v1.layers.dense(
    to_tensor_2d,
    num_attention_heads * size_per_head,
    activation=value_act,
    name="value",
    kernel_initializer=create_initializer(initializer_range))
```
to
```python
query_layer = tf.compat.v1.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="q_lin",
      kernel_initializer=create_initializer(initializer_range))

# `key_layer` = [B*T, N*H]
key_layer = tf.compat.v1.layers.dense(
    to_tensor_2d,
    num_attention_heads * size_per_head,
    activation=key_act,
    name="k_lin",
    kernel_initializer=create_initializer(initializer_range))

# `value_layer` = [B*T, N*H]
value_layer = tf.compat.v1.layers.dense(
    to_tensor_2d,
    num_attention_heads * size_per_head,
    activation=value_act,
    name="v_lin",
    kernel_initializer=create_initializer(initializer_range))
```
to align with the pre-trained Huggingface model.

* Changed scope in transformer_model function:
```python
for layer_idx in range(num_hidden_layers):
  with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
    layer_input = prev_output

    with tf.compat.v1.variable_scope("attention"):
      attention_heads = []
      with tf.compat.v1.variable_scope("self"):
        attention_head = attention_layer(
            from_tensor=layer_input,
            to_tensor=layer_input,
            attention_mask=attention_mask,
            num_attention_heads=num_attention_heads,
            size_per_head=attention_head_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=True,
            batch_size=batch_size,
            from_seq_length=seq_length,
            to_seq_length=seq_length)
        attention_heads.append(attention_head)

      attention_output = None
      if len(attention_heads) == 1:
        attention_output = attention_heads[0]
      else:
        # In the case where we have other sequences, we just concatenate
        # them to the self-attention head before the projection.
        attention_output = tf.concat(attention_heads, axis=-1)

      # Run a linear projection of `hidden_size` then add a residual
      # with `layer_input`.
      with tf.compat.v1.variable_scope("output"):
        attention_output = tf.compat.v1.layers.dense(
            attention_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        attention_output = dropout(attention_output, hidden_dropout_prob)
        attention_output = layer_norm(attention_output + layer_input)

    # The activation is only applied to the "intermediate" hidden layer.
    with tf.compat.v1.variable_scope("intermediate"):
      intermediate_output = tf.compat.v1.layers.dense(
          attention_output,
          intermediate_size,
          activation=intermediate_act_fn,
          kernel_initializer=create_initializer(initializer_range))

    # Down-project back to `hidden_size` then add the residual.
    with tf.compat.v1.variable_scope("output"):
      layer_output = tf.compat.v1.layers.dense(
          intermediate_output,
          hidden_size,
          kernel_initializer=create_initializer(initializer_range))
      layer_output = dropout(layer_output, hidden_dropout_prob)
      layer_output = layer_norm(layer_output + attention_output)
      prev_output = layer_output
      all_layer_outputs.append(layer_output)
```
to
```python
for layer_idx in range(num_hidden_layers):
  with tf.compat.v1.variable_scope("layer/%d" % layer_idx):
    layer_input = prev_output

    with tf.compat.v1.variable_scope("attention"):
      attention_heads = []
      attention_head = attention_layer(
          from_tensor=layer_input,
          to_tensor=layer_input,
          attention_mask=attention_mask,
          num_attention_heads=num_attention_heads,
          size_per_head=attention_head_size,
          attention_probs_dropout_prob=attention_probs_dropout_prob,
          initializer_range=initializer_range,
          do_return_2d_tensor=True,
          batch_size=batch_size,
          from_seq_length=seq_length,
          to_seq_length=seq_length)
      attention_heads.append(attention_head)

      attention_output = None
      if len(attention_heads) == 1:
        attention_output = attention_heads[0]
      else:
        # In the case where we have other sequences, we just concatenate
        # them to the self-attention head before the projection.
        attention_output = tf.concat(attention_heads, axis=-1)

      # Run a linear projection of `hidden_size` then add a residual
      # with `layer_input`.
      attention_output = tf.compat.v1.layers.dense(
          attention_output,
          hidden_size,
          name="out_lin",
          kernel_initializer=create_initializer(initializer_range))
      attention_output = dropout(attention_output, hidden_dropout_prob)
    attention_output = layer_norm(attention_output + layer_input, name='sa_layer_norm')

    # The activation is only applied to the "intermediate" hidden layer.
    with tf.compat.v1.variable_scope("ffn"):
      intermediate_output = tf.compat.v1.layers.dense(
          attention_output,
          intermediate_size,
          activation=intermediate_act_fn,
          kernel_initializer=create_initializer(initializer_range),
          name='lin1')

      # Down-project back to `hidden_size` then add the residual.
      layer_output = tf.compat.v1.layers.dense(
          intermediate_output,
          hidden_size,
          kernel_initializer=create_initializer(initializer_range),
          name='lin2')
      layer_output = dropout(layer_output, hidden_dropout_prob)

    layer_output = layer_norm(layer_output + attention_output, name="output_layer_norm")
    prev_output = layer_output
    all_layer_outputs.append(layer_output)
```
to align with the pre-trained Huggingface model.

* In get_assignment_map_from_checkpoint function:
```python
def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue

    tvar = name_to_variable[name]
    assert is_resource_variable(tvar)
    assignment_map[name] = tvar
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)
```
to
```python
def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  suffix = '/.ATTRIBUTES/VARIABLE_VALUE'
  for x in init_vars:
    (name, var) = (x[0], x[1])
    temp_name = name
    if name.endswith(suffix):
      temp_name = name[:-len(suffix)]
    if temp_name not in name_to_variable:
      continue

    tvar = name_to_variable[temp_name]
    assert is_resource_variable(tvar)
    assignment_map[name] = tvar
    initialized_variable_names[temp_name] = 1
    initialized_variable_names[temp_name + ":0"] = 1

  return (assignment_map, initialized_variable_names)
```
to ensure correct variable initialization from the Huggingface pre-trained model.

### optimization.py

* Changed
```python
optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
```
to
```python
optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.0,
        beta_1=0.8,
        beta_2=0.95,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
```
to get better accuracy (EM/F1) for DistilBERT.

### run_squad.py

* Changed `bert_config_file` to `distilbert_config_file`

### distilbert_squad_main.py

* Added distilbert config and downloading pre-trained model in build_command function:
```python
# Write distilbert config jdon file to path
bcfg_path = str(pretrained_model_path.joinpath("bert_config.json"))
bert_config = json.load(open(bcfg_path, 'r'))
bert_config["type_vocab_size"] = 16
bert_config["num_hidden_layers"] = 6
distilbcfg_path = str(pretrained_model_path.joinpath("distilbert_config.json"))
distilbert_file = open(distilbcfg_path, "w")
json.dump(bert_config, distilbert_file)
distilbert_file.close()

# Download huggingface pretrained distilbert_base_uncased model
if not os.path.isfile(str(pretrained_model_path.joinpath("distilbert-base-uncased.ckpt-1.index"))):
    model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-uncased')
    model.compile()
    ckpt_prefix = os.path.join(pretrained_model_path, 'distilbert-base-uncased.ckpt')
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.save(file_prefix=ckpt_prefix)
print ("Reusing existing pre-trained model 'distilbert-base-uncased'")
ic_path = str(pretrained_model_path.joinpath("distilbert-base-uncased.ckpt-1"))
```

### download_pretrained_model.py

* Added download_pretrained_model_distilbert function:
```python
def download_pretrained_model_distilbert(pretrained_model_bert, pretrained_model_distilbert='distilbert-base-uncased'):
  host_name = socket.gethostname()
  this_dir = os.getcwd()
  pretrained_model_bert_path = Path(pretrained_model_bert)
  try:
      # Write distilbert config jdon file to path
      print(f"{host_name}: *** Generating distilbert_config.json...")
      bcfg_path = str(pretrained_model_bert_path.joinpath("bert_config.json"))
      bert_config = json.load(open(bcfg_path, 'r'))
      bert_config["type_vocab_size"] = 16
      bert_config["num_hidden_layers"] = 6
      distilbcfg_path = str(pretrained_model_bert_path.joinpath("distilbert_config.json"))
      distilbert_file = open(distilbcfg_path, "w")
      json.dump(bert_config, distilbert_file)
      distilbert_file.close()

      # Download pre-trained distilbert model
      if not os.path.isfile(str(pretrained_model_bert_path.joinpath("distilbert-base-uncased.ckpt-1.index"))):
          print(f"{host_name}: *** Downloading pre-trained distilbert model...")
          model = TFDistilBertForTokenClassification.from_pretrained(pretrained_model_distilbert)
          model.compile()
          ckpt_prefix = os.path.join(pretrained_model_bert_path, 'distilbert-base-uncased.ckpt')
          checkpoint = tf.train.Checkpoint(model=model)
          checkpoint.save(file_prefix=ckpt_prefix)
      else:
          print(f"{host_name}: *** Reusing existing pre-trained model 'distilbert-base-uncased'")
  except Exception as exc:
      os.chdir(this_dir)
      raise Exception(f"{host_name}: Error in {__file__} download_pretrained_model_distilbert()") from exc
```