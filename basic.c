#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_VOCAB_SIZE 500

typedef struct
{
    int first;
    int second;
} Pair;

typedef struct
{
    Pair *pairs;
    int *counts;
    int size;
    int capacity;
} PairCounts;

typedef struct
{
    unsigned char **vocab; // Dynamic array of tokens
    int vocab_size;
    PairCounts merges; // To store merges as a dictionary of pairs to int
} BasicTokenizer;

void init_pair_counts(PairCounts *counts)
{
    counts->pairs = NULL;
    counts->counts = NULL;
    counts->size = 0;
    counts->capacity = 0;
}

/**
 * Adds a pair count to the PairCounts structure or increments the count
 * if the pair already exists.
 *
 * This function searches for the given pair in the PairCounts structure. If found,
 * it increments the existing count by the specified initial_count. If the pair
 * is not found, the pair and the initial_count are added to the structure.
 * If necessary, the function reallocates memory to expand the PairCounts arrays
 * when the current capacity is reached.
 *
 * @param counts A pointer to the PairCounts structure where the pair and count are to be added.
 * @param pair The pair of integers (defined in a Pair structure) to be added or updated.
 * @param initial_count The count to be added for the pair. If the pair exists, this value
 *                      is added to the existing count.
 */
void add_pair_count(PairCounts *counts, Pair pair, int initial_count)
{
    for (int i = 0; i < counts->size; i++)
    {
        if (counts->pairs[i].first == pair.first && counts->pairs[i].second == pair.second)
        {
            counts->counts[i] += initial_count;
            return;
        }
    }
    if (counts->size == counts->capacity)
    {
        counts->capacity = counts->capacity == 0 ? 4 : counts->capacity * 2;
        counts->pairs = realloc(counts->pairs, counts->capacity * sizeof(Pair));
        counts->counts = realloc(counts->counts, counts->capacity * sizeof(int));
    }
    counts->pairs[counts->size] = pair;
    counts->counts[counts->size] = initial_count;
    counts->size++;
}

/**
 * Generates a PairCounts structure containing counts of consecutive integer pairs
 * in the provided array. This function processes an array of integers and counts
 * how many times each consecutive pair (n, n+1) appears.
 *
 * The function initializes a PairCounts structure, iterates through the array of
 * integers, and for each consecutive pair of elements, it updates or adds a new
 * entry in the PairCounts structure using the add_pair_count function.
 *
 * @param ids An array of integers for which consecutive pairs are to be counted.
 * @param length The number of elements in the ids array.
 * @return A PairCounts structure populated with pairs and their corresponding counts.
 *
 * Example usage:
 * int ids[] = {1, 2, 1, 2, 3};
 * int length = 5;
 * PairCounts counts = get_stats(ids, length);
 * // The counts structure now contains the frequency of each consecutive pair in ids.
 */
PairCounts get_stats(int *ids, int length)
{
    PairCounts counts;
    init_pair_counts(&counts);
    for (int i = 0; i < length - 1; i++)
    {
        Pair pair = {ids[i], ids[i + 1]};
        add_pair_count(&counts, pair, 1);
    }
    return counts;
}

/**
 * Merges consecutive pairs in an integer array that match a specified pair, replacing them
 * with a single specified index. This function is typically used in tokenization processes
 * where frequent pairs of tokens (or characters) are merged into a single new token to
 * compress the sequence or to prepare for further processing.
 *
 * The function allocates a new integer array to store the result of the merge. For each
 * pair of elements in the input array that matches the specified pair, the function inserts
 * the new index (`idx`) into the new array instead of the pair. Elements not part of a matching
 * pair are copied directly to the new array.
 *
 * @param ids Pointer to the original array of integers.
 * @param length The number of elements in the `ids` array.
 * @param pair The pair of integers to search for in the `ids` array. Consecutive elements
 *             matching this pair are replaced in the output array.
 * @param idx The new integer value that replaces each matching pair in the output array.
 * @param new_length Pointer to an integer where the function will store the length of the
 *                   new array.
 * @return Pointer to the new dynamically allocated array containing the merged integers.
 *
 * Example usage:
 * int ids[] = {1, 2, 3, 1, 2};
 * int length = 5;
 * Pair pair = {1, 2};
 * int new_length;
 * int* merged_ids = merge(ids, length, pair, 99, &new_length);
 * // `merged_ids` will contain: {99, 3, 99}, `new_length` will be set to 3
 */
int *merge(int *ids, int length, Pair pair, int idx, int *new_length)
{
    int *newids = malloc(length * sizeof(int));
    int j = 0;
    for (int i = 0; i < length; i++)
    {
        if (i < length - 1 && ids[i] == pair.first && ids[i + 1] == pair.second)
        {
            newids[j++] = idx;
            i++; // Skip the next element
        }
        else
        {
            newids[j++] = ids[i];
        }
    }
    *new_length = j;
    return newids;
}

/**
 * Creates and initializes a new BasicTokenizer instance. This function allocates memory
 * for a BasicTokenizer structure and initializes its components, specifically the vocabulary
 * and merges structure. The vocabulary is initialized with INITIAL_VOCAB_SIZE tokens where
 * each token is a single character from the ASCII set. This setup is typical for byte-level
 * tokenization where each byte (character) in the input text can be directly mapped to a token.
 *
 * The function also initializes the merges structure, which is used to store merged token pairs
 * and their frequencies as the tokenizer processes text data.
 *
 * @return Pointer to the newly created BasicTokenizer structure.
 *
 * Example usage:
 * BasicTokenizer* tokenizer = create_basic_tokenizer();
 * // `tokenizer` can now be used for tokenization tasks.
 */
BasicTokenizer *create_basic_tokenizer()
{
    BasicTokenizer *tokenizer = malloc(sizeof(BasicTokenizer));
    tokenizer->vocab = malloc(INITIAL_VOCAB_SIZE * sizeof(unsigned char *));
    for (int i = 0; i < INITIAL_VOCAB_SIZE; i++)
    {
        tokenizer->vocab[i] = malloc(2);
        tokenizer->vocab[i][0] = i;
        tokenizer->vocab[i][1] = '\0';
    }
    tokenizer->vocab_size = INITIAL_VOCAB_SIZE;
    init_pair_counts(&tokenizer->merges);
    return tokenizer;
}

/**
 * Trains the BasicTokenizer by processing the given text to identify and merge
 * frequent pairs of characters (or tokens). This function adapts the Byte Pair Encoding
 * algorithm to progressively merge the most frequent adjacent pairs of tokens into
 * single tokens, thereby increasing the tokenizer's vocabulary based on the text input.
 *
 * The function first converts the input text into an array of integers (`ids`), where
 * each integer represents the ASCII value of a character in the text. It then performs
 * a series of merges, each time finding the most frequent pair of tokens and replacing
 * all occurrences of that pair in the text with a new token. Each new token is added
 * to the tokenizer's vocabulary.
 *
 * @param tokenizer Pointer to the BasicTokenizer instance to be trained.
 * @param text Unsigned char array containing the input text to be processed.
 * @param vocab_size The desired size of the vocabulary after training. The number of
 *                   merges performed is determined by the difference between `vocab_size`
 *                   and `INITIAL_VOCAB_SIZE`.
 * @param verbose If non-zero, the function prints detailed logs of each merge operation,
 *                showing progress and statistics such as which pairs were merged and
 *                the number of occurrences.
 *
 * Example usage:
 * BasicTokenizer* tokenizer = create_basic_tokenizer();
 * unsigned char text[] = "example text for tokenizer training";
 * train(tokenizer, text, 300, 1);  // Trains tokenizer to expand its vocab to 300 tokens
 */
void train(BasicTokenizer *tokenizer, unsigned char *text, int vocab_size, int verbose)
{
    int text_length = strlen((char *)text);
    int *ids = malloc(text_length * sizeof(int));
    for (int i = 0; i < text_length; i++)
    {
        ids[i] = text[i];
    }

    int num_merges = vocab_size - INITIAL_VOCAB_SIZE;
    PairCounts stats;
    for (int i = 0; i < num_merges; i++)
    {
        stats = get_stats(ids, text_length);
        int max_idx = 0;
        for (int j = 1; j < stats.size; j++)
        {
            if (stats.counts[j] > stats.counts[max_idx])
            {
                max_idx = j;
            }
        }
        Pair max_pair = stats.pairs[max_idx];
        int new_idx = INITIAL_VOCAB_SIZE + i;
        int new_length;
        ids = merge(ids, text_length, max_pair, new_idx, &new_length);
        text_length = new_length;
        add_pair_count(&tokenizer->merges, max_pair, new_idx);
        tokenizer->vocab = realloc(tokenizer->vocab, (tokenizer->vocab_size + 1) * sizeof(unsigned char *));
        tokenizer->vocab[tokenizer->vocab_size] = malloc(3); // Assuming new tokens are two chars long
        tokenizer->vocab[tokenizer->vocab_size][0] = max_pair.first;
        tokenizer->vocab[tokenizer->vocab_size][1] = max_pair.second;
        tokenizer->vocab[tokenizer->vocab_size][2] = '\0';
        tokenizer->vocab_size++;
        if (verbose)
        {
            printf("merge %d/%d: (%d, %d) -> %d had %d occurrences\n", i + 1, num_merges, max_pair.first, max_pair.second, new_idx, stats.counts[max_idx]);
        }
    }
    free(ids);
}

/**
 * Decodes an array of integer IDs back into text using a BasicTokenizer's vocabulary. This function
 * assumes that each integer in the `ids` array corresponds to an index in the tokenizer's vocabulary,
 * where each index maps to a specific token (or character). The function iterates through the `ids`
 * array, retrieves the corresponding tokens from the tokenizer's vocabulary, and prints them to
 * form the decoded text string.
 *
 * This is typically used after text has been encoded into token IDs and some processing has been done,
 * allowing the original or modified text to be reconstructed from the token IDs.
 *
 * @param tokenizer A pointer to the BasicTokenizer structure which contains the vocabulary
 *                  used for decoding the integer IDs.
 * @param ids Pointer to an integer array containing the token IDs to be decoded.
 * @param length The number of elements in the `ids` array, indicating how many tokens to decode.
 *
 * Example usage:
 * BasicTokenizer* tokenizer = create_basic_tokenizer();  // Assumes tokenizer is already trained
 * int ids[] = {72, 101, 108, 108, 111};  // Example token IDs for 'Hello'
 * int length = sizeof(ids) / sizeof(ids[0]);
 * decode(tokenizer, ids, length);  // Output will be 'Hello'
 */
void decode(BasicTokenizer *tokenizer, int *ids, int length)
{
    if (tokenizer == NULL || ids == NULL)
    {
        fprintf(stderr, "Invalid input: tokenizer and ids must not be NULL.\n");
        return;
    }
    printf("Decoded text: ");
    for (int i = 0; i < length; i++)
    {
        // Check if the ID is within the range of the vocabulary size
        if (ids[i] < 0 || ids[i] >= tokenizer->vocab_size)
        {
            fprintf(stderr, "Error: ID %d out of range (0-%d).\n", ids[i], tokenizer->vocab_size - 1);
            continue;
        }
        printf("%s", tokenizer->vocab[ids[i]]);
    }
    printf("\n");
}

/**
 * Encodes the given text into an array of integer IDs, where each ID represents the ASCII value
 * of the corresponding character in the text. This simple encoding mechanism converts each character
 * of the text into its ASCII equivalent and stores it in an integer array, essentially transforming
 * the input string into a sequence of integers for further processing or tokenization.
 *
 * The function also returns the length of the encoded array, which is equal to the length of the input text.
 *
 * @param tokenizer A pointer to the BasicTokenizer structure. While this structure is part of the function's
 *                  signature, it's not utilized in this implementation but could be used in future for more
 *                  complex encoding schemes that involve a tokenizer's vocabulary.
 * @param text Unsigned char array representing the input text to be encoded.
 * @param length Pointer to an integer where the function will store the length of the output array.
 *               This value will be set to the number of characters in the input text.
 * @return Pointer to an integer array containing the ASCII values of the characters in the input text.
 *
 * Example usage:
 * BasicTokenizer* tokenizer = create_basic_tokenizer();
 * unsigned char text[] = "hello";
 * int length;
 * int* encoded_ids = encode(tokenizer, text, &length);
 * for (int i = 0; i < length; i++) {
 *     printf("%d ", encoded_ids[i]);
 * }
 * printf("\n");
 * free(encoded_ids);
 */
int *encode(BasicTokenizer *tokenizer, unsigned char *text, int *length)
{
    int text_length = strlen((char *)text);
    int *ids = malloc(text_length * sizeof(int));
    for (int i = 0; i < text_length; i++)
    {
        ids[i] = text[i];
    }

    *length = text_length;
    return ids;
}

void test_tokenizer(BasicTokenizer *tokenizer, unsigned char **input_texts, int num_texts)
{
    for (int t = 0; t < num_texts; t++)
    {
        // Encode the input text
        int encoded_length;
        int *encoded_ids = encode(tokenizer, input_texts[t], &encoded_length);

        // Print the encoded IDs
        printf("Input text: \"%s\"\n", input_texts[t]);
        printf("Encoded IDs: ");
        for (int i = 0; i < encoded_length; i++)
        {
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

void cleanup_tokenizer(BasicTokenizer *tokenizer)
{
    // Free each string in the vocabulary
    for (int i = 0; i < tokenizer->vocab_size; i++)
    {
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

int main()
{
    // Example text to train the tokenizer
    unsigned char text[] = "hello world of machine learning beautiful you are there";

    // Create and train the tokenizer
    BasicTokenizer *tokenizer = create_basic_tokenizer();
    int target_vocab_size = INITIAL_VOCAB_SIZE + 10; // Adjust as needed
    train(tokenizer, text, target_vocab_size, 1);    // Verbose output enabled

    // Array of example texts to process
    unsigned char *test_texts[] = {
        (unsigned char *)"hello machine",
        (unsigned char *)"machine learning",
        (unsigned char *)"world learning hello",
        (unsigned char *)"beautiful hello",
        (unsigned char *)"you there"};
    int num_texts = sizeof(test_texts) / sizeof(test_texts[0]);

    // Process each text: encode and decode
    test_tokenizer(tokenizer, test_texts, num_texts);

    // Cleanup all resources used by the tokenizer
    cleanup_tokenizer(tokenizer);

    return 0;
}
