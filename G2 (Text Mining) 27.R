# install packages
install.packages("tm")
install.packages("topicmodels")
install.packages("reshape2")
install.packages("ggplot2")
install.packages("wordcloud")
install.packages("pals")
install.packages("SnowballC")
install.packages("lda")
install.packages("ldatuning")
install.packages("tidyverse")
install.packages("kableExtra")
install.packages("DT")
install.packages("flextable")
install.packages("remotes")
install.packages("devtools")
remotes::install_github("rlesur/klippy")
install.packages("randomForest")
install.packages("arules")
install.packages("arulesViz")
install.packages("cluster")
install.packages(c("plotly"))



# load packages
library(knitr)
library(plotly)
library(cluster)
library(randomForest)
library(kableExtra)
library(tidyverse)
library(arules)
library(arulesViz)
library(dplyr)
library(DT)
library(tm)
library(topicmodels)
library(reshape2)
library(ggplot2)
library(wordcloud)
library(pals)
library(SnowballC)
library(lda)
library(ldatuning)
library(flextable)
library(wordcloud)
library(RColorBrewer)
library(dplyr)


# Specify the file path
file_path <- "journal_data.csv"

# Read the CSV file
data <- read.csv(file_path, na.strings = c("", "NA", "N/A"))

# Remove rows with NA values
data <- na.omit(data)

# Print the first few rows of the data to check if it's loaded correctly
head(data)

# Create a corpus from the abstracts
corpus <- Corpus(VectorSource(data$abstract))

# Preprocess the corpus
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)

# Create a Document-Term Matrix (DTM) for Bag of Words
dtm <- DocumentTermMatrix(corpus)

# Visualize the distribution of popular words using a Word Cloud
word_freq <- colSums(as.matrix(dtm))
top_words <- head(sort(word_freq, decreasing = TRUE), 200)  # Adjust 50 to the desired number of top words

# custom colors
custom_colors <- c("#001219", "#005F73", "#0A9396", "#94D2BD", "#E9D8A6", "#EE9B00", "#CA6702", "#BB3E03")

wordcloud(words = names(top_words), freq = top_words, scale = c(3, 0.5), random.order = FALSE, colors = custom_colors)


#########   TF-idf

# Transform the Document-Term Matrix (DTM) into a TF-IDF (Term Frequency-Inverse Document Frequency) matrix
tfidf_dtm <- weightTfIdf(dtm, normalize = TRUE)

# Inspect the resulting TF-IDF matrix
inspect(tfidf_dtm)

# Extract the term frequency (TF) for document 10
tf_doc10 <- as.matrix(dtm[10,])
tf <- colSums(tf_doc10)

# Extract the TF-IDF for document 10
tfidf_doc10 <- as.matrix(tfidf_dtm[10,])
tfidf <- colSums(tfidf_doc10)

# custom colors
custom_colors <- c("#001219", "#005F73", "#0A9396", "#94D2BD", "#E9D8A6", "#EE9B00", "#CA6702", "#BB3E03")

# plotting area
par(mfrow = c(2, 1), mar = c(4, 4, 2, 2))

# bar plot for the top 20 terms based on TF in document 10
barplot(sort(tf, decreasing = TRUE)[1:20], col = custom_colors, las = 2, main = "Top 20 Terms by TF", xlab = "Term", ylab = "Frequency", cex.names = 0.7)

# bar plot for the top 20 terms based on TF-IDF in document 10
barplot(sort(tfidf, decreasing = TRUE)[1:20], col = custom_colors, las = 2, main = "Top 20 Terms by TF-IDF", xlab = "Term", ylab = "TF-IDF Weight", cex.names = 0.7)

######  Yearly Trends

# Metrics of interest (e.g., 'views', 'citations')
metrics_of_interest <- c("views", "citations")  # Add more metrics if needed

# Set custom colors for different journals
journal_colors <- c("#001219", "#005F73")

# line plot for each metric across different years
par(mfrow = c(length(metrics_of_interest), 1), mar = c(4, 4, 2, 2))

for (i in 1:length(metrics_of_interest)) {
  metric_of_interest <- metrics_of_interest[i]
  plot(data$year, data[[metric_of_interest]], type = "b", pch = 16, col = journal_colors[i], xlab = "Year", ylab = metric_of_interest, main = paste("Yearly Trends in", metric_of_interest), cex.main = 0.8)
}

# Get the coordinates for placing the legend outside the plot
legend_x <- max(data$year) + 1
legend_y <- max(data[[metrics_of_interest[1]]])  # Adjust as needed

# Add a legend outside the plot
legend(x = legend_x, y = legend_y, legend = unique(data$journal), col = journal_colors[1:length(unique(data$journal))], pch = 16, title = "Journal", bty = "n", cex = 0.8)

#################  LDA

# Set minimum frequency for terms
minimumFrequency <- 5

# Compute Document Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus, control = list(bounds = list(global = c(minimumFrequency, Inf))))

# Remove empty rows from DTM
sel_idx <- slam::row_sums(dtm) > 0
dtm1 <- dtm[sel_idx, ]
data <- data[sel_idx, ]

# Find optimal number of topics
result <- ldatuning::FindTopicsNumber(
  dtm1,
  topics = seq(from = 2, to = 20, by = 1),
  metrics = c("CaoJuan2009", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  verbose = TRUE
)

# Visualize optimal number of topics
FindTopicsNumber_plot(result)

# Number of topics
K <- 20

# Set random number generator seed
set.seed(9161)

# Compute the LDA model
topicModel <- LDA(dtm1, K, method = "Gibbs", control = list(iter = 500, verbose = 25))

# Extract some results from the model
tmResult <- posterior(topicModel)
beta <- tmResult$terms
theta <- tmResult$topics

# Display information about the model
nTerms(dtm1)  # lengthOfVocab
dim(beta)    # K distributions over nTerms(dtm1) terms
rowSums(beta)  # Rows in beta sum to 1
nDocs(dtm1)  # Size of the collection

# Example term data
exampleTermData <- terms(topicModel, 10)
exampleTermData[, 1:8]

# Top 5 terms per topic
top5termsPerTopic <- terms(topicModel, 5)
topicNames <- apply(top5termsPerTopic, 2, paste, collapse = " ")

# Example document IDs
exampleIds <- c(2, 100, 200)
lapply(corpus[exampleIds], as.character)

# Display partial content of example documents
print(paste0(exampleIds[1], ": ", substr(content(corpus[[exampleIds[1]]]), 0, 400), '...'))

# Visualization of topic proportions in example documents
N <- length(exampleIds)
topicProportionExamples <- theta[exampleIds, ]
colnames(topicProportionExamples) <- topicNames
vizDataFrame <- melt(cbind(data.frame(topicProportionExamples), document = factor(1:N)), variable.name = "topic", id.vars = "document")  
ggplot(data = vizDataFrame, aes(topic, value, fill = document), ylab = "proportion") + 
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +  
  coord_flip() +
  facet_wrap(~ document, ncol = N)

# Display alpha from the previous model
attr(topicModel, "alpha")

# Create a new LDA model with a specified alpha value
topicModel2 <- LDA(dtm1, K, method = "Gibbs", control = list(iter = 500, verbose = 25, alpha = 0.2))
tmResult <- posterior(topicModel2)
theta <- tmResult$topics
beta <- tmResult$terms
topicNames <- apply(terms(topicModel2, 5), 2, paste, collapse = " ")

# Re-rank top topic terms for topic names
topicNames <- apply(lda::top.topic.words(beta, 5, by.score = T), 2, paste, collapse = " ")

# Most probable topics in the entire collection
topicProportions <- colSums(theta) / nDocs(dtm1)
names(topicProportions) <- topicNames
sort(topicProportions, decreasing = TRUE)

# Print sorted topic proportions
soP <- sort(topicProportions, decreasing = TRUE)
paste(round(soP, 5), ":", names(soP))

# Filter documents based on a specific topic and threshold
topicToFilter <- grep('children', topicNames)[1]
topicThreshold <- 0.2
selectedDocumentIndexes <- which(theta[, topicToFilter] >= topicThreshold)
filteredCorpus <- corpus[selectedDocumentIndexes]

# Display length of filtered corpus
filteredCorpus

# Get mean topic proportions per year and visualize
topic_proportion_per_year <- aggregate(theta, by = list(year = data$year), mean)
colnames(topic_proportion_per_year)[2:(K + 1)] <- topicNames
vizDataFrame <- melt(topic_proportion_per_year, id.vars = "year")
ggplot(vizDataFrame, aes(x = year, y = value, fill = variable)) + 
  geom_bar(stat = "identity") + ylab("proportion") + 
  scale_fill_manual(values = paste0(alphabet(20), "FF"), name = "year") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


######  Regression Model

# Check for missing values and remove rows with NA in 'citations' column
data1 <- data %>% drop_na(citations)

# Create a training set and a testing set (80% training, 20% testing)
set.seed(123)  # for reproducibility
indices <- sample(1:nrow(data1), 0.8 * nrow(data1))
train_data <- data1[indices, ]
test_data <- data1[-indices, ]

# Build a linear regression model
lm_model <- lm(citations ~ views + altmetric, data = train_data)

# Summary of the regression model
summary(lm_model)

#######################Classification

# Convert 'journal' into a factor
data$journal <- as.factor(data$journal)

# Split the data into training and testing sets
set.seed(123)  # for reproducibility
train_indices <- sample(seq_len(nrow(data)), 0.7 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Convert 'journal' to factor
train_data$journal <- as.factor(train_data$journal)

# Train the RandomForest model
rf_model <- randomForest(journal ~ ., data = train_data, ntree = 100)

# Make predictions on the test set
predictions <- predict(rf_model, newdata = test_data)

# Evaluate the accuracy
accuracy <- sum(predictions == test_data$journal) / nrow(test_data)
cat("Accuracy:", accuracy, "\n")


####################Association

# Convert DTM to a binary matrix
dtm_binary <- as(dtm, "matrix")
dtm_binary[dtm_binary > 0] <- 1

# Convert to transactions
transactions <- as(dtm_binary, "transactions")

# Mine association rules
rules <- apriori(data, parameter = list(support = 0.1, confidence = 0.8))

sorted_rules <- sort(rules, by = "lift")
top_rules <- head(sorted_rules, 200)

# Plot association rules
plot(top_rules, method = "graph")

# column containing abstracts
transactions <- tm::Corpus(VectorSource(data$abstract))

# Convert to a matrix for the arules package
trans_matrix <- tm::DocumentTermMatrix(transactions)
trans_matrix <- as(as.matrix(trans_matrix), "transactions")

association_rules <- apriori(trans_matrix, parameter = list(support = 0.1, confidence = 0.5))
summary(association_rules)

# Customize colors
plot_colors <- c("#0A9396","#005F73","#EE9B00")

# Scatterplot of support vs. confidence
plot(association_rules, method = "scatterplot", measure = c("support", "confidence"), col = plot_colors)

# Graph of the rules
plot(association_rules, method = "graph", control = list(type = "items"), col = plot_colors)

# Matrix-based plot
plot(association_rules, method = "matrix", measure = c("support", "confidence"), col = plot_colors)


######## Clustering

# Replace 'my_corpus' with your actual corpus
corpus <- Corpus(VectorSource(data$abstract))
# Create a document-term matrix
dtm <- DocumentTermMatrix(corpus)

# Convert the document-term matrix to a matrix
mat <- as.matrix(dtm)

# Perform k-means clustering (you can choose the number of clusters 'k')
k <- 3  # Change the number of clusters as needed
kmeans_result <- kmeans(mat, centers = k)

# Assign cluster labels to each document
cluster_labels <- kmeans_result$cluster

# Calculate TF-IDF scores using the tm package
dtm_tfidf <- weightTfIdf(dtm)

# Combine cluster labels and TF-IDF scores
cluster_tfidf <- cbind(cluster = cluster_labels, as.data.frame(as.matrix(dtm_tfidf)))

# For each cluster, identify top words by looking at features with the highest mean TF-IDF scores
top_words_per_cluster <- lapply(1:k, function(i) {
  cluster_data <- cluster_tfidf[cluster_tfidf$cluster == i, -1, drop = FALSE]
  top_words_indices <- order(rowMeans(cluster_data, na.rm = TRUE), decreasing = TRUE)
  top_words <- colnames(mat)[top_words_indices]
  return(top_words)
})

# Print or use the top words for each cluster
for (i in 1:k) {
  cat("Cluster", i, ":", paste(top_words_per_cluster[[i]], collapse = ", "), "\n")
}


#########   PCA

# Assuming 'mat' is your data matrix

# Set seed for reproducibility
set.seed(123)

# Create a random subset of your data matrix
sample_indices <- sample(1:nrow(mat), size = 1000)  # Adjust the size as needed
subset_mat <- mat[sample_indices, ]

# Exclude columns with zero variance
non_zero_var_cols <- apply(subset_mat, 2, var) > 0
subset_mat <- subset_mat[, non_zero_var_cols]

# Check if there are still columns left after excluding zero variance columns
if (ncol(subset_mat) > 1) {
  # Perform PCA on the subset
  pca_result <- prcomp(subset_mat, center = TRUE, scale. = TRUE)
  pca_components <- pca_result$x[, 1:2]
  
  # Assuming 'kmeans_result' contains the result of k-means clustering
  cluster_labels <- factor(kmeans_result$cluster[sample_indices])  # Adjust accordingly
  
  # Create a data frame with the PCA components and cluster labels
  pca_data <- data.frame(PC1 = pca_components[, 1], PC2 = pca_components[, 2], Cluster = cluster_labels)
  
  # Create a data frame with the PCA components
  pca_data <- data.frame(PC1 = pca_components[, 1], PC2 = pca_components[, 2])
  
  # Plot the PCA components using plot_ly for interactive plotting
  plot_ly(data = pca_data, x = ~PC1, y = ~PC2, type = "scatter", mode = "markers", marker = list(size = 5)) %>%
    layout(title = "PCA Visualization",
           xaxis = list(title = "Principal Component 1"),
           yaxis = list(title = "Principal Component 2"))
} else {
  cat("No columns with non-zero variance for PCA.\n")
}
