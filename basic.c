#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define INITIAL_VOCAB_SIZE 500

typedef struct {
    int first;
    int second;
} Pair;

typedef struct {
    Pair* pairs;
    int* counts;
    int size;
    int capacity;
} PairCounts;

typedef struct {
    unsigned char** vocab;  // Dynamic array of tokens
    int vocab_size;
    PairCounts merges;      // To store merges as a dictionary of pairs to int
} BasicTokenizer;


void init_pair_counts(PairCounts* counts) {
    counts->pairs = NULL;
    counts->counts = NULL;
    counts->size = 0;
    counts->capacity = 0;
}

void add_pair_count(PairCounts* counts, Pair pair, int initial_count) {
    for (int i = 0; i < counts->size; i++) {
        if (counts->pairs[i].first == pair.first && counts->pairs[i].second == pair.second) {
            counts->counts[i] += initial_count;
            return;
        }
    }
    if (counts->size == counts->capacity) {
        counts->capacity = counts->capacity == 0 ? 4 : counts->capacity * 2;
        counts->pairs = realloc(counts->pairs, counts->capacity * sizeof(Pair));
        counts->counts = realloc(counts->counts, counts->capacity * sizeof(int));
    }
    counts->pairs[counts->size] = pair;
    counts->counts[counts->size] = initial_count;
    counts->size++;
}

PairCounts get_stats(int* ids, int length) {
    PairCounts counts;
    init_pair_counts(&counts);
    for (int i = 0; i < length - 1; i++) {
        Pair pair = {ids[i], ids[i + 1]};
        add_pair_count(&counts, pair, 1);
    }
    return counts;
}

int* merge(int* ids, int length, Pair pair, int idx, int* new_length) {
    int* newids = malloc(length * sizeof(int));
    int j = 0;
    for (int i = 0; i < length; i++) {
        if (i < length - 1 && ids[i] == pair.first && ids[i + 1] == pair.second) {
            newids[j++] = idx;
            i++;  // Skip the next element
        } else {
            newids[j++] = ids[i];
        }
    }
    *new_length = j;
    return newids;
}

BasicTokenizer* create_basic_tokenizer() {
    BasicTokenizer* tokenizer = malloc(sizeof(BasicTokenizer));
    tokenizer->vocab = malloc(INITIAL_VOCAB_SIZE * sizeof(unsigned char*));
    for (int i = 0; i < INITIAL_VOCAB_SIZE; i++) {
        tokenizer->vocab[i] = malloc(2);
        tokenizer->vocab[i][0] = i;
        tokenizer->vocab[i][1] = '\0';
    }
    tokenizer->vocab_size = INITIAL_VOCAB_SIZE;
    init_pair_counts(&tokenizer->merges);
    return tokenizer;
}

void train(BasicTokenizer* tokenizer, unsigned char* text, int vocab_size, int verbose) {
    int text_length = strlen((char *)text);
    int* ids = malloc(text_length * sizeof(int));
    for (int i = 0; i < text_length; i++) {
        ids[i] = text[i];
    }

    int num_merges = vocab_size - INITIAL_VOCAB_SIZE;
    PairCounts stats;
    for (int i = 0; i < num_merges; i++) {
        stats = get_stats(ids, text_length);
        int max_idx = 0;
        for (int j = 1; j < stats.size; j++) {
            if (stats.counts[j] > stats.counts[max_idx]) {
                max_idx = j;
            }
        }
        Pair max_pair = stats.pairs[max_idx];
        int new_idx = INITIAL_VOCAB_SIZE + i;
        int new_length;
        ids = merge(ids, text_length, max_pair, new_idx, &new_length);
        text_length = new_length;
        add_pair_count(&tokenizer->merges, max_pair, new_idx);
        tokenizer->vocab = realloc(tokenizer->vocab, (tokenizer->vocab_size + 1) * sizeof(unsigned char*));
        tokenizer->vocab[tokenizer->vocab_size] = malloc(3); // Assuming new tokens are two chars long
        tokenizer->vocab[tokenizer->vocab_size][0] = max_pair.first;
        tokenizer->vocab[tokenizer->vocab_size][1] = max_pair.second;
        tokenizer->vocab[tokenizer->vocab_size][2] = '\0';
        tokenizer->vocab_size++;
        if (verbose) {
            printf("merge %d/%d: (%d, %d) -> %d had %d occurrences\n", i + 1, num_merges, max_pair.first, max_pair.second, new_idx, stats.counts[max_idx]);
        }
    }
    free(ids);
}

void decode(BasicTokenizer* tokenizer, int* ids, int length) {
    for (int i = 0; i < length; i++) {
        printf("%s", tokenizer->vocab[ids[i]]);
    }
    printf("\n");
}

int* encode(BasicTokenizer* tokenizer, unsigned char* text, int* length) {
    int text_length = strlen((char *)text);
    int* ids = malloc(text_length * sizeof(int));
    for (int i = 0; i < text_length; i++) {
        ids[i] = text[i];
    }

    *length = text_length;
    return ids;
}



void test_tokenizer(BasicTokenizer* tokenizer, unsigned char** input_texts, int num_texts) {
    for (int t = 0; t < num_texts; t++) {
        // Encode the input text
        int encoded_length;
        int* encoded_ids = encode(tokenizer, input_texts[t], &encoded_length);

        // Print the encoded IDs
        printf("Input text: \"%s\"\n", input_texts[t]);
        printf("Encoded IDs: ");
        for (int i = 0; i < encoded_length; i++) {
            printf("%d ", encoded_ids[i]);
        }
        printf("\n");

        // Decode the encoded IDs and print the result
        printf("Decoded text: ");
        decode(tokenizer, encoded_ids, encoded_length);
        printf("\n\n");

        // Free the encoded ids array
        free(encoded_ids);
    }
}

void cleanup_tokenizer(BasicTokenizer* tokenizer) {
    // Free each string in the vocabulary
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->vocab[i]);
    }

    // Free the vocab array itself
    free(tokenizer->vocab);

    // Free the merges structure
    free(tokenizer->merges.pairs);
    free(tokenizer->merges.counts);

    // Finally, free the tokenizer structure
    free(tokenizer);
}

int main() {
    // Example text to train the tokenizer
    unsigned char text[] = "hello world of machine learning beautiful you are there";

    // Create and train the tokenizer
    BasicTokenizer* tokenizer = create_basic_tokenizer();
    int target_vocab_size = INITIAL_VOCAB_SIZE + 10;  // Adjust as needed
    train(tokenizer, text, target_vocab_size, 1);  // Verbose output enabled

    // Array of example texts to process
    unsigned char* test_texts[] = {
        (unsigned char*)"hello machine",
        (unsigned char*)"machine learning",
        (unsigned char*)"world learning hello",
        (unsigned char*)"beautiful hello",
        (unsigned char*)"you there"
    };
    int num_texts = sizeof(test_texts) / sizeof(test_texts[0]);

    // Process each text: encode and decode
    test_tokenizer(tokenizer, test_texts, num_texts);

    // Cleanup all resources used by the tokenizer
    cleanup_tokenizer(tokenizer);

    return 0;
}
