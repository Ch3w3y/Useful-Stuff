# ===============================================================================
# TEXT MINING AND NLP IN R - COMPREHENSIVE TEMPLATE
# ===============================================================================
# Complete guide to text processing, sentiment analysis, topic modeling, and NLP
# Author: Data Science Toolkit
# Last Updated: 2024
# ===============================================================================

# Required Libraries
suppressMessages({
  library(tm)
  library(tidytext)
  library(dplyr)
  library(stringr)
  library(ggplot2)
  library(wordcloud)
  library(wordcloud2)
  library(RColorBrewer)
  library(topicmodels)
  library(LDAvis)
  library(servr)
  library(textdata)
  library(syuzhet)
  library(sentimentr)
  library(vader)
  library(tidyr)
  library(igraph)
  library(ggraph)
  library(widyr)
  library(SnowballC)
  library(textstem)
  library(textclean)
  library(quanteda)
  library(quanteda.textplots)
  library(quanteda.textstats)
  library(spacyr)
  library(text2vec)
  library(glmnet)
  library(randomForest)
  library(caret)
  library(e1071)
  library(cluster)
  library(factoextra)
  library(plotly)
  library(DT)
  library(kableExtra)
})

# ===============================================================================
# 1. TEXT DATA LOADING AND PREPROCESSING
# ===============================================================================

#' Load Text Data from Various Sources
#' @param source Data source type
#' @param path File path or connection details
load_text_data <- function(source = "sample", path = NULL) {
  cat("Loading text data from:", source, "\n")
  
  if (source == "sample") {
    # Create sample text data
    sample_texts <- c(
      "I love this product! It's amazing and works perfectly.",
      "This service is terrible. I'm very disappointed.",
      "The weather today is absolutely wonderful and sunny.",
      "Machine learning is revolutionizing data science.",
      "Natural language processing helps computers understand text.",
      "The movie was boring and I fell asleep halfway through.",
      "Excellent customer service! They were very helpful.",
      "The food at this restaurant is delicious and fresh.",
      "I'm excited about the new AI developments in healthcare.",
      "The book was well-written but too long for my taste.",
      "Climate change is a serious global challenge we must address.",
      "The software update fixed many bugs but introduced new ones.",
      "I highly recommend this hotel for business travelers.",
      "The concert was loud but the music was fantastic.",
      "Online education has become increasingly important during the pandemic."
    )
    
    text_data <- data.frame(
      id = 1:length(sample_texts),
      text = sample_texts,
      category = sample(c("positive", "negative", "neutral"), length(sample_texts), replace = TRUE),
      stringsAsFactors = FALSE
    )
    
  } else if (source == "csv") {
    text_data <- read.csv(path, stringsAsFactors = FALSE)
    
  } else if (source == "txt") {
    # Read text file
    raw_text <- readLines(path)
    text_data <- data.frame(
      id = 1:length(raw_text),
      text = raw_text,
      stringsAsFactors = FALSE
    )
    
  } else if (source == "twitter") {
    # Placeholder for Twitter API integration
    cat("Twitter API integration not implemented in this template\n")
    return(NULL)
  }
  
  cat("Loaded", nrow(text_data), "text documents\n")
  return(text_data)
}

#' Comprehensive Text Preprocessing
#' @param text_data Data frame with text column
#' @param text_col Name of text column
#' @param steps Preprocessing steps to apply
preprocess_text <- function(text_data, text_col = "text", 
                           steps = c("lowercase", "remove_punct", "remove_numbers", 
                                   "remove_stopwords", "stem")) {
  cat("\n=== TEXT PREPROCESSING ===\n")
  
  # Create a copy
  processed_data <- text_data
  original_text <- processed_data[[text_col]]
  
  cat("Original text sample:\n")
  cat(paste(substr(original_text[1], 1, 100), "...\n"))
  
  # Apply preprocessing steps
  processed_text <- original_text
  
  if ("lowercase" %in% steps) {
    processed_text <- tolower(processed_text)
    cat("✓ Converted to lowercase\n")
  }
  
  if ("remove_punct" %in% steps) {
    processed_text <- str_replace_all(processed_text, "[[:punct:]]", " ")
    cat("✓ Removed punctuation\n")
  }
  
  if ("remove_numbers" %in% steps) {
    processed_text <- str_replace_all(processed_text, "[[:digit:]]", "")
    cat("✓ Removed numbers\n")
  }
  
  if ("remove_whitespace" %in% steps) {
    processed_text <- str_squish(processed_text)
    cat("✓ Cleaned whitespace\n")
  }
  
  if ("remove_stopwords" %in% steps) {
    # Remove English stopwords
    stopwords_list <- stopwords("english")
    processed_text <- sapply(processed_text, function(x) {
      words <- unlist(strsplit(x, "\\s+"))
      words <- words[!words %in% stopwords_list]
      paste(words, collapse = " ")
    })
    cat("✓ Removed stopwords\n")
  }
  
  if ("stem" %in% steps) {
    processed_text <- sapply(processed_text, function(x) {
      words <- unlist(strsplit(x, "\\s+"))
      words <- wordStem(words, language = "english")
      paste(words, collapse = " ")
    })
    cat("✓ Applied stemming\n")
  }
  
  if ("lemmatize" %in% steps) {
    processed_text <- sapply(processed_text, function(x) {
      words <- unlist(strsplit(x, "\\s+"))
      words <- textstem::lemmatize_words(words)
      paste(words, collapse = " ")
    })
    cat("✓ Applied lemmatization\n")
  }
  
  processed_data$processed_text <- as.character(processed_text)
  
  cat("\nProcessed text sample:\n")
  cat(paste(substr(processed_data$processed_text[1], 1, 100), "...\n"))
  
  return(processed_data)
}

#' Create Document-Term Matrix
#' @param text_data Preprocessed text data
#' @param text_col Name of text column
create_dtm <- function(text_data, text_col = "processed_text") {
  cat("\n=== CREATING DOCUMENT-TERM MATRIX ===\n")
  
  # Using tm package
  corpus <- Corpus(VectorSource(text_data[[text_col]]))
  
  # Create DTM
  dtm <- DocumentTermMatrix(corpus, 
                           control = list(bounds = list(global = c(2, Inf)),
                                        weighting = weightTf))
  
  cat("DTM dimensions:", dim(dtm), "\n")
  cat("Sparsity:", dtm$dimnames$Docs %>% length(), "documents,", 
      dtm$dimnames$Terms %>% length(), "terms\n")
  
  # Remove sparse terms (appear in < 5% of documents)
  dtm_sparse <- removeSparseTerms(dtm, 0.95)
  cat("After removing sparse terms:", dim(dtm_sparse), "\n")
  
  # Most frequent terms
  freq <- colSums(as.matrix(dtm_sparse))
  freq_sorted <- sort(freq, decreasing = TRUE)
  
  cat("\nTop 10 most frequent terms:\n")
  print(head(freq_sorted, 10))
  
  return(list(
    dtm = dtm,
    dtm_sparse = dtm_sparse,
    term_freq = freq_sorted
  ))
}

# ===============================================================================
# 2. EXPLORATORY TEXT ANALYSIS
# ===============================================================================

#' Text Statistics and Exploration
#' @param text_data Text data frame
#' @param text_col Name of text column
explore_text_data <- function(text_data, text_col = "text") {
  cat("\n=== TEXT EXPLORATION ===\n")
  
  # Basic statistics
  text_stats <- text_data %>%
    mutate(
      char_count = nchar(.data[[text_col]]),
      word_count = str_count(.data[[text_col]], "\\S+"),
      sentence_count = str_count(.data[[text_col]], "[.!?]+"),
      avg_word_length = char_count / word_count
    )
  
  cat("TEXT STATISTICS:\n")
  cat("- Total documents:", nrow(text_stats), "\n")
  cat("- Average characters per document:", round(mean(text_stats$char_count, na.rm = TRUE), 1), "\n")
  cat("- Average words per document:", round(mean(text_stats$word_count, na.rm = TRUE), 1), "\n")
  cat("- Average sentences per document:", round(mean(text_stats$sentence_count, na.rm = TRUE), 1), "\n")
  
  # Distribution plots
  p1 <- ggplot(text_stats, aes(x = word_count)) +
    geom_histogram(bins = 20, fill = "skyblue", alpha = 0.7) +
    labs(title = "Distribution of Word Counts", x = "Word Count", y = "Frequency") +
    theme_minimal()
  
  p2 <- ggplot(text_stats, aes(x = char_count)) +
    geom_histogram(bins = 20, fill = "lightcoral", alpha = 0.7) +
    labs(title = "Distribution of Character Counts", x = "Character Count", y = "Frequency") +
    theme_minimal()
  
  print(p1)
  print(p2)
  
  return(text_stats)
}

#' Word Frequency Analysis
#' @param text_data Text data frame
#' @param text_col Name of text column
#' @param top_n Number of top words to return
analyze_word_frequency <- function(text_data, text_col = "processed_text", top_n = 20) {
  cat("\n=== WORD FREQUENCY ANALYSIS ===\n")
  
  # Tokenize and count words
  word_freq <- text_data %>%
    select(id = 1, text = all_of(text_col)) %>%
    unnest_tokens(word, text) %>%
    count(word, sort = TRUE)
  
  cat("Total unique words:", nrow(word_freq), "\n")
  cat("Total word occurrences:", sum(word_freq$n), "\n")
  
  # Top words
  top_words <- head(word_freq, top_n)
  
  cat("\nTop", top_n, "most frequent words:\n")
  print(top_words)
  
  # Visualization
  p1 <- top_words %>%
    ggplot(aes(x = reorder(word, n), y = n)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = paste("Top", top_n, "Most Frequent Words"),
         x = "Word", y = "Frequency") +
    theme_minimal()
  
  print(p1)
  
  # Word cloud
  if (nrow(word_freq) > 10) {
    wordcloud(words = word_freq$word, freq = word_freq$n, 
              max.words = 100, random.order = FALSE, 
              colors = brewer.pal(8, "Dark2"))
  }
  
  return(word_freq)
}

#' N-gram Analysis
#' @param text_data Text data frame
#' @param text_col Name of text column
#' @param n N-gram size (2 for bigrams, 3 for trigrams)
analyze_ngrams <- function(text_data, text_col = "processed_text", n = 2) {
  cat("\n=== N-GRAM ANALYSIS (n =", n, ") ===\n")
  
  ngram_name <- paste0("ngram_", n)
  
  # Create n-grams
  ngrams <- text_data %>%
    select(id = 1, text = all_of(text_col)) %>%
    unnest_tokens(!!ngram_name, text, token = "ngrams", n = n) %>%
    filter(!is.na(.data[[ngram_name]])) %>%
    count(.data[[ngram_name]], sort = TRUE)
  
  cat("Total unique", n, "-grams:", nrow(ngrams), "\n")
  
  # Top n-grams
  top_ngrams <- head(ngrams, 15)
  
  cat("\nTop 15 most frequent", n, "-grams:\n")
  print(top_ngrams)
  
  # Visualization
  p1 <- top_ngrams %>%
    ggplot(aes(x = reorder(.data[[ngram_name]], n), y = n)) +
    geom_col(fill = "darkgreen") +
    coord_flip() +
    labs(title = paste("Top", n, "-grams"),
         x = paste(n, "-gram"), y = "Frequency") +
    theme_minimal()
  
  print(p1)
  
  return(ngrams)
}

# ===============================================================================
# 3. SENTIMENT ANALYSIS
# ===============================================================================

#' Comprehensive Sentiment Analysis
#' @param text_data Text data frame
#' @param text_col Name of text column
#' @param methods Sentiment analysis methods to use
analyze_sentiment <- function(text_data, text_col = "text", 
                            methods = c("afinn", "bing", "nrc", "vader", "syuzhet")) {
  cat("\n=== SENTIMENT ANALYSIS ===\n")
  
  sentiment_results <- text_data
  
  # 1. AFINN Lexicon (-5 to +5)
  if ("afinn" %in% methods) {
    cat("Applying AFINN sentiment analysis...\n")
    
    afinn_scores <- text_data %>%
      select(id = 1, text = all_of(text_col)) %>%
      unnest_tokens(word, text) %>%
      inner_join(get_sentiments("afinn"), by = "word") %>%
      group_by(id) %>%
      summarise(sentiment_afinn = sum(value), .groups = "drop")
    
    sentiment_results <- sentiment_results %>%
      left_join(afinn_scores, by = c("1" = "id")) %>%
      mutate(sentiment_afinn = ifelse(is.na(sentiment_afinn), 0, sentiment_afinn))
  }
  
  # 2. Bing Lexicon (positive/negative)
  if ("bing" %in% methods) {
    cat("Applying Bing sentiment analysis...\n")
    
    bing_scores <- text_data %>%
      select(id = 1, text = all_of(text_col)) %>%
      unnest_tokens(word, text) %>%
      inner_join(get_sentiments("bing"), by = "word") %>%
      count(id, sentiment) %>%
      pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
      mutate(sentiment_bing = positive - negative)
    
    sentiment_results <- sentiment_results %>%
      left_join(bing_scores[c("id", "sentiment_bing")], by = c("1" = "id")) %>%
      mutate(sentiment_bing = ifelse(is.na(sentiment_bing), 0, sentiment_bing))
  }
  
  # 3. NRC Emotion Lexicon
  if ("nrc" %in% methods) {
    cat("Applying NRC emotion analysis...\n")
    
    nrc_emotions <- text_data %>%
      select(id = 1, text = all_of(text_col)) %>%
      unnest_tokens(word, text) %>%
      inner_join(get_sentiments("nrc"), by = "word") %>%
      count(id, sentiment) %>%
      pivot_wider(names_from = sentiment, values_from = n, values_fill = 0, names_prefix = "nrc_")
    
    sentiment_results <- sentiment_results %>%
      left_join(nrc_emotions, by = c("1" = "id"))
  }
  
  # 4. VADER Sentiment
  if ("vader" %in% methods && requireNamespace("vader", quietly = TRUE)) {
    cat("Applying VADER sentiment analysis...\n")
    
    vader_scores <- sapply(text_data[[text_col]], vader_df)
    sentiment_results$sentiment_vader <- vader_scores["compound", ]
  }
  
  # 5. Syuzhet Package
  if ("syuzhet" %in% methods) {
    cat("Applying Syuzhet sentiment analysis...\n")
    
    sentiment_results$sentiment_syuzhet <- get_sentiment(text_data[[text_col]], method = "syuzhet")
  }
  
  # Summary statistics
  sentiment_cols <- grep("sentiment_", names(sentiment_results), value = TRUE)
  
  if (length(sentiment_cols) > 0) {
    cat("\n=== SENTIMENT SUMMARY ===\n")
    for (col in sentiment_cols) {
      if (col %in% names(sentiment_results)) {
        values <- sentiment_results[[col]]
        values <- values[!is.na(values)]
        
        cat(col, ":\n")
        cat("  Mean:", round(mean(values), 3), "\n")
        cat("  SD:", round(sd(values), 3), "\n")
        cat("  Range:", round(range(values), 3), "\n")
        
        # Classification
        if (grepl("afinn|bing|syuzhet|vader", col)) {
          positive <- sum(values > 0)
          negative <- sum(values < 0)
          neutral <- sum(values == 0)
          
          cat("  Positive:", positive, "documents\n")
          cat("  Negative:", negative, "documents\n")
          cat("  Neutral:", neutral, "documents\n")
        }
        cat("\n")
      }
    }
    
    # Visualization
    if ("sentiment_afinn" %in% names(sentiment_results)) {
      p1 <- ggplot(sentiment_results, aes(x = sentiment_afinn)) +
        geom_histogram(bins = 20, fill = "lightblue", alpha = 0.7) +
        geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
        labs(title = "Distribution of AFINN Sentiment Scores", 
             x = "Sentiment Score", y = "Count") +
        theme_minimal()
      
      print(p1)
    }
  }
  
  return(sentiment_results)
}

#' Emotion Analysis
#' @param text_data Text data frame
#' @param text_col Name of text column
analyze_emotions <- function(text_data, text_col = "text") {
  cat("\n=== EMOTION ANALYSIS ===\n")
  
  # Using NRC emotion lexicon
  emotions <- text_data %>%
    select(id = 1, text = all_of(text_col)) %>%
    unnest_tokens(word, text) %>%
    inner_join(get_sentiments("nrc"), by = "word") %>%
    filter(!sentiment %in% c("positive", "negative")) %>%
    count(sentiment, sort = TRUE)
  
  cat("Emotion distribution across all documents:\n")
  print(emotions)
  
  # Visualization
  p1 <- emotions %>%
    ggplot(aes(x = reorder(sentiment, n), y = n, fill = sentiment)) +
    geom_col() +
    coord_flip() +
    scale_fill_viridis_d() +
    labs(title = "Emotion Distribution", x = "Emotion", y = "Word Count") +
    theme_minimal() +
    theme(legend.position = "none")
  
  print(p1)
  
  # Document-level emotions
  doc_emotions <- text_data %>%
    select(id = 1, text = all_of(text_col)) %>%
    unnest_tokens(word, text) %>%
    inner_join(get_sentiments("nrc"), by = "word") %>%
    filter(!sentiment %in% c("positive", "negative")) %>%
    count(id, sentiment) %>%
    pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)
  
  return(list(
    overall_emotions = emotions,
    document_emotions = doc_emotions
  ))
}

# ===============================================================================
# 4. TOPIC MODELING
# ===============================================================================

#' Latent Dirichlet Allocation (LDA) Topic Modeling
#' @param text_data Text data frame
#' @param text_col Name of text column
#' @param k Number of topics
#' @param method LDA method
perform_lda_analysis <- function(text_data, text_col = "processed_text", k = 5, method = "Gibbs") {
  cat("\n=== LDA TOPIC MODELING ===\n")
  
  # Create DTM
  corpus <- Corpus(VectorSource(text_data[[text_col]]))
  dtm <- DocumentTermMatrix(corpus, control = list(bounds = list(global = c(2, Inf))))
  
  # Remove empty documents
  row_sums <- apply(dtm, 1, sum)
  dtm_clean <- dtm[row_sums > 0, ]
  
  cat("Running LDA with", k, "topics on", nrow(dtm_clean), "documents...\n")
  
  # Fit LDA model
  lda_model <- LDA(dtm_clean, k = k, method = method, 
                   control = list(seed = 123))
  
  # Extract topics
  topics <- tidy(lda_model, matrix = "beta")
  
  # Top terms per topic
  top_terms <- topics %>%
    group_by(topic) %>%
    slice_max(beta, n = 10) %>%
    ungroup() %>%
    arrange(topic, -beta)
  
  cat("\nTop terms per topic:\n")
  for (i in 1:k) {
    topic_terms <- top_terms %>% filter(topic == i)
    cat("Topic", i, ":", paste(topic_terms$term[1:5], collapse = ", "), "\n")
  }
  
  # Document-topic probabilities
  doc_topics <- tidy(lda_model, matrix = "gamma")
  
  # Visualization
  p1 <- top_terms %>%
    mutate(term = reorder_within(term, beta, topic)) %>%
    ggplot(aes(beta, term, fill = factor(topic))) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~ topic, scales = "free") +
    scale_y_reordered() +
    labs(title = "Top Terms per Topic", x = "Beta (word probability)", y = "Terms") +
    theme_minimal()
  
  print(p1)
  
  # Topic coherence and perplexity
  perplexity_score <- perplexity(lda_model, dtm_clean)
  cat("\nModel perplexity:", round(perplexity_score, 2), "\n")
  
  return(list(
    model = lda_model,
    topics = topics,
    top_terms = top_terms,
    doc_topics = doc_topics,
    perplexity = perplexity_score
  ))
}

#' Topic Model Evaluation and Selection
#' @param text_data Text data frame
#' @param text_col Name of text column
#' @param k_range Range of k values to test
evaluate_topic_models <- function(text_data, text_col = "processed_text", k_range = 2:10) {
  cat("\n=== TOPIC MODEL EVALUATION ===\n")
  
  # Prepare DTM
  corpus <- Corpus(VectorSource(text_data[[text_col]]))
  dtm <- DocumentTermMatrix(corpus, control = list(bounds = list(global = c(2, Inf))))
  row_sums <- apply(dtm, 1, sum)
  dtm_clean <- dtm[row_sums > 0, ]
  
  # Evaluate different k values
  results <- data.frame(
    k = integer(),
    perplexity = numeric(),
    log_likelihood = numeric()
  )
  
  cat("Evaluating models for k =", paste(k_range, collapse = ", "), "\n")
  
  for (k in k_range) {
    cat("Testing k =", k, "... ")
    
    lda_model <- LDA(dtm_clean, k = k, method = "Gibbs", 
                     control = list(seed = 123, iter = 100))
    
    perp <- perplexity(lda_model, dtm_clean)
    loglik <- logLik(lda_model)
    
    results <- rbind(results, data.frame(
      k = k,
      perplexity = perp,
      log_likelihood = as.numeric(loglik)
    ))
    
    cat("Done\n")
  }
  
  # Find optimal k
  optimal_k <- results[which.min(results$perplexity), "k"]
  cat("\nOptimal number of topics (lowest perplexity):", optimal_k, "\n")
  
  # Visualization
  p1 <- ggplot(results, aes(x = k, y = perplexity)) +
    geom_line(color = "blue") +
    geom_point(color = "red") +
    geom_vline(xintercept = optimal_k, linetype = "dashed", color = "green") +
    labs(title = "Topic Model Evaluation", 
         x = "Number of Topics (k)", y = "Perplexity") +
    theme_minimal()
  
  print(p1)
  
  return(list(
    results = results,
    optimal_k = optimal_k
  ))
}

# ===============================================================================
# 5. TEXT CLASSIFICATION
# ===============================================================================

#' Text Classification with Machine Learning
#' @param text_data Text data frame with labels
#' @param text_col Name of text column
#' @param label_col Name of label column
#' @param methods Classification methods to use
classify_text <- function(text_data, text_col = "processed_text", label_col = "category", 
                         methods = c("nb", "svm", "rf")) {
  cat("\n=== TEXT CLASSIFICATION ===\n")
  
  # Prepare data
  if (!label_col %in% names(text_data)) {
    cat("Label column not found. Cannot perform classification.\n")
    return(NULL)
  }
  
  # Create DTM
  corpus <- Corpus(VectorSource(text_data[[text_col]]))
  dtm <- DocumentTermMatrix(corpus, control = list(bounds = list(global = c(2, Inf))))
  
  # Convert to matrix
  dtm_matrix <- as.matrix(dtm)
  
  # Prepare labels
  labels <- factor(text_data[[label_col]])
  
  cat("Classification task:", length(levels(labels)), "classes\n")
  cat("Class distribution:\n")
  print(table(labels))
  
  # Split data
  set.seed(123)
  train_idx <- createDataPartition(labels, p = 0.8, list = FALSE)
  
  train_x <- dtm_matrix[train_idx, ]
  train_y <- labels[train_idx]
  test_x <- dtm_matrix[-train_idx, ]
  test_y <- labels[-train_idx]
  
  cat("\nTraining set:", length(train_y), "documents\n")
  cat("Test set:", length(test_y), "documents\n")
  
  # Classification results
  results <- list()
  
  # 1. Naive Bayes
  if ("nb" %in% methods) {
    cat("\nTraining Naive Bayes classifier...\n")
    
    nb_model <- naiveBayes(train_x, train_y)
    nb_pred <- predict(nb_model, test_x)
    nb_conf <- confusionMatrix(nb_pred, test_y)
    
    results$naive_bayes <- list(
      model = nb_model,
      predictions = nb_pred,
      confusion_matrix = nb_conf,
      accuracy = nb_conf$overall["Accuracy"]
    )
    
    cat("Naive Bayes Accuracy:", round(nb_conf$overall["Accuracy"], 3), "\n")
  }
  
  # 2. SVM
  if ("svm" %in% methods) {
    cat("\nTraining SVM classifier...\n")
    
    svm_model <- svm(train_x, train_y, kernel = "linear")
    svm_pred <- predict(svm_model, test_x)
    svm_conf <- confusionMatrix(svm_pred, test_y)
    
    results$svm <- list(
      model = svm_model,
      predictions = svm_pred,
      confusion_matrix = svm_conf,
      accuracy = svm_conf$overall["Accuracy"]
    )
    
    cat("SVM Accuracy:", round(svm_conf$overall["Accuracy"], 3), "\n")
  }
  
  # 3. Random Forest
  if ("rf" %in% methods) {
    cat("\nTraining Random Forest classifier...\n")
    
    # Reduce features for Random Forest (memory intensive)
    important_features <- head(names(sort(colSums(train_x), decreasing = TRUE)), 100)
    train_x_rf <- train_x[, important_features]
    test_x_rf <- test_x[, important_features]
    
    rf_model <- randomForest(train_x_rf, train_y, ntree = 100)
    rf_pred <- predict(rf_model, test_x_rf)
    rf_conf <- confusionMatrix(rf_pred, test_y)
    
    results$random_forest <- list(
      model = rf_model,
      predictions = rf_pred,
      confusion_matrix = rf_conf,
      accuracy = rf_conf$overall["Accuracy"]
    )
    
    cat("Random Forest Accuracy:", round(rf_conf$overall["Accuracy"], 3), "\n")
  }
  
  # Model comparison
  if (length(results) > 1) {
    cat("\n=== MODEL COMPARISON ===\n")
    accuracies <- sapply(results, function(x) x$accuracy)
    best_model <- names(accuracies)[which.max(accuracies)]
    
    cat("Best model:", best_model, "with accuracy:", round(max(accuracies), 3), "\n")
  }
  
  return(results)
}

# ===============================================================================
# 6. NETWORK ANALYSIS AND WORD ASSOCIATIONS
# ===============================================================================

#' Word Co-occurrence Network Analysis
#' @param text_data Text data frame
#' @param text_col Name of text column
#' @param min_count Minimum word count threshold
analyze_word_networks <- function(text_data, text_col = "processed_text", min_count = 2) {
  cat("\n=== WORD NETWORK ANALYSIS ===\n")
  
  # Create word co-occurrence matrix
  word_cors <- text_data %>%
    select(id = 1, text = all_of(text_col)) %>%
    unnest_tokens(word, text) %>%
    add_count(word) %>%
    filter(n >= min_count) %>%
    select(-n) %>%
    pairwise_cor(word, id, sort = TRUE)
  
  cat("Found", nrow(word_cors), "word pair correlations\n")
  
  # Filter strong correlations
  strong_cors <- word_cors %>%
    filter(correlation > 0.25)
  
  cat("Strong correlations (> 0.25):", nrow(strong_cors), "\n")
  
  if (nrow(strong_cors) > 0) {
    # Create network graph
    word_graph <- strong_cors %>%
      graph_from_data_frame()
    
    # Plot network
    set.seed(123)
    p1 <- ggraph(word_graph, layout = "fr") +
      geom_edge_link(aes(edge_alpha = correlation), show.legend = FALSE) +
      geom_node_point(color = "lightblue", size = 3) +
      geom_node_text(aes(label = name), repel = TRUE, size = 3) +
      theme_void() +
      labs(title = "Word Co-occurrence Network")
    
    print(p1)
  }
  
  return(list(
    correlations = word_cors,
    strong_correlations = strong_cors,
    graph = if (nrow(strong_cors) > 0) word_graph else NULL
  ))
}

# ===============================================================================
# 7. COMPREHENSIVE WORKFLOW
# ===============================================================================

#' Complete Text Mining Workflow
#' @param data_source Data source
#' @param data_path Path to data file
complete_text_workflow <- function(data_source = "sample", data_path = NULL) {
  cat("===============================================================================\n")
  cat("COMPLETE TEXT MINING AND NLP WORKFLOW\n")
  cat("===============================================================================\n")
  
  # 1. Load data
  cat("\n1. LOADING DATA...\n")
  text_data <- load_text_data(data_source, data_path)
  
  if (is.null(text_data)) {
    cat("Failed to load data. Exiting.\n")
    return(NULL)
  }
  
  # 2. Preprocess text
  cat("\n2. PREPROCESSING TEXT...\n")
  processed_data <- preprocess_text(text_data, "text")
  
  # 3. Exploratory analysis
  cat("\n3. EXPLORATORY ANALYSIS...\n")
  text_stats <- explore_text_data(processed_data, "text")
  word_freq <- analyze_word_frequency(processed_data, "processed_text")
  bigrams <- analyze_ngrams(processed_data, "processed_text", n = 2)
  
  # 4. Sentiment analysis
  cat("\n4. SENTIMENT ANALYSIS...\n")
  sentiment_data <- analyze_sentiment(processed_data, "text")
  emotions <- analyze_emotions(processed_data, "text")
  
  # 5. Topic modeling
  cat("\n5. TOPIC MODELING...\n")
  # First evaluate optimal number of topics
  topic_eval <- evaluate_topic_models(processed_data, "processed_text", k_range = 2:5)
  optimal_k <- min(topic_eval$optimal_k, 5)  # Cap at 5 for demo
  
  # Run LDA with optimal k
  lda_results <- perform_lda_analysis(processed_data, "processed_text", k = optimal_k)
  
  # 6. Text classification (if labels available)
  classification_results <- NULL
  if ("category" %in% names(processed_data)) {
    cat("\n6. TEXT CLASSIFICATION...\n")
    classification_results <- classify_text(processed_data, "processed_text", "category")
  }
  
  # 7. Word networks
  cat("\n7. WORD NETWORK ANALYSIS...\n")
  network_results <- analyze_word_networks(processed_data, "processed_text")
  
  # 8. Generate comprehensive report
  cat("\n8. GENERATING REPORT...\n")
  generate_text_report(processed_data, word_freq, sentiment_data, lda_results)
  
  # Return all results
  return(list(
    original_data = text_data,
    processed_data = processed_data,
    text_stats = text_stats,
    word_frequency = word_freq,
    ngrams = bigrams,
    sentiment = sentiment_data,
    emotions = emotions,
    topics = lda_results,
    classification = classification_results,
    networks = network_results
  ))
}

#' Generate Text Analysis Report
#' @param text_data Processed text data
#' @param word_freq Word frequency results
#' @param sentiment_data Sentiment analysis results
#' @param topic_results Topic modeling results
generate_text_report <- function(text_data, word_freq, sentiment_data, topic_results) {
  cat("===============================================================================\n")
  cat("TEXT MINING AND NLP ANALYSIS REPORT\n")
  cat("===============================================================================\n")
  
  # Basic statistics
  cat("\nDATA OVERVIEW:\n")
  cat("- Total documents:", nrow(text_data), "\n")
  cat("- Total unique words:", nrow(word_freq), "\n")
  cat("- Average words per document:", round(mean(str_count(text_data$text, "\\S+"), na.rm = TRUE), 1), "\n")
  
  # Top words
  cat("\nTOP 10 WORDS:\n")
  top_words_formatted <- paste(head(word_freq$word, 10), collapse = ", ")
  cat(top_words_formatted, "\n")
  
  # Sentiment summary
  sentiment_cols <- grep("sentiment_", names(sentiment_data), value = TRUE)
  if (length(sentiment_cols) > 0) {
    cat("\nSENTIMENT SUMMARY:\n")
    
    if ("sentiment_afinn" %in% names(sentiment_data)) {
      afinn_scores <- sentiment_data$sentiment_afinn
      positive_pct <- round(sum(afinn_scores > 0) / length(afinn_scores) * 100, 1)
      negative_pct <- round(sum(afinn_scores < 0) / length(afinn_scores) * 100, 1)
      neutral_pct <- round(sum(afinn_scores == 0) / length(afinn_scores) * 100, 1)
      
      cat("- Positive documents:", positive_pct, "%\n")
      cat("- Negative documents:", negative_pct, "%\n")
      cat("- Neutral documents:", neutral_pct, "%\n")
    }
  }
  
  # Topic summary
  if (!is.null(topic_results)) {
    cat("\nTOPIC MODELING SUMMARY:\n")
    cat("- Number of topics:", max(topic_results$top_terms$topic), "\n")
    cat("- Model perplexity:", round(topic_results$perplexity, 2), "\n")
    
    # Show top term for each topic
    for (i in 1:max(topic_results$top_terms$topic)) {
      top_term <- topic_results$top_terms %>% 
        filter(topic == i) %>% 
        slice_max(beta, n = 1) %>% 
        pull(term)
      cat("- Topic", i, "top term:", top_term, "\n")
    }
  }
}

# ===============================================================================
# EXAMPLE USAGE
# ===============================================================================

# Run complete analysis
# results <- complete_text_workflow("sample")
# 
# # Load your own text data
# my_results <- complete_text_workflow("csv", "path/to/your/text_data.csv")
# 
# # Individual analyses
# text_data <- load_text_data("sample")
# processed <- preprocess_text(text_data)
# sentiment <- analyze_sentiment(processed)
# topics <- perform_lda_analysis(processed, k = 3)

cat("Text Mining and NLP Template Loaded Successfully!\n")
cat("Key Functions Available:\n")
cat("- complete_text_workflow(): Run full analysis pipeline\n")
cat("- preprocess_text(): Clean and prepare text data\n")
cat("- analyze_sentiment(): Multi-method sentiment analysis\n")
cat("- perform_lda_analysis(): Topic modeling with LDA\n")
cat("- classify_text(): Text classification with ML\n")
cat("- analyze_word_networks(): Word co-occurrence networks\n") 