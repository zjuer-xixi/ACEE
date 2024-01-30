#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#define K1 1 		// Define a constant value K1 for the number of iterations of algorithm1
#define K2 100		// Define a constant value K2 for the number of iterations of algorithm2
#define K3 1000		// Define a constant value K2 for the number of iterations of algorithm3

clock_t start, stop; // Initialize variables for time measurement

// Function to print the maximum submatrix
void print(int **matrix, int start_row, int start_column, int end_row, int end_column, int N, int x);

// Algorithm 1 for finding the maximum matrix sum
int max_matrix_sum1(int **matrix, int N);

// Algorithm 2 for finding the maximum matrix sum
int max_matrix_sum2(int **matrix, int N);

// Algorithm 3 for finding the maximum matrix sum
int max_matrix_sum3(int **matrix, int N);

// Main function
int main() {
	// Define the size of the matrix
	int N;
	scanf("%d", &N);

	int matrix[N][N], max1, max2, max3;

	// Set the random seed to generate different random numbers each time the program runs
	srand(time(NULL));

	// Display the original matrix
	printf("Original matrix is:\n");

	// Initialize the matrix
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			matrix[i][j] = rand() % 201 - 100;
			printf("%4d ", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	// Call the functions to find the maximum matrix sum using different algorithms
	max1 = max_matrix_sum1((int **)matrix, N);
	max2 = max_matrix_sum2((int **)matrix, N);
	max3 = max_matrix_sum3((int **)matrix, N);

	// Print the maximum values obtained by each algorithm
	printf("Maximum value obtained by Algorithm 1: %d\n", max1);
	printf("Maximum value obtained by Algorithm 2: %d\n", max2);
	printf("Maximum value obtained by Algorithm 3: %d\n", max3);

	system("pause");
	return 0;
}

// Function to find the maximum matrix sum using the first algorithm
int max_matrix_sum1(int **matrix, int N) {
	start = clock(); // Record the starting time of the algorithm

	int sum, max_sum, start_row, start_column, end_row, end_column, i1, j1, i2, j2, m, n;//Define the necessary variables
	int iteration;

	// Iterate the algorithm for K times
	for(iteration = 1; iteration <= K1; iteration++) {

		max_sum = 0;
		start_row = start_column = end_row = end_column = 0;

		// Nested loops to iterate through the matrix and find the maximum submatrix sum
		for(i1 = 0; i1 < N; i1++) {
			for(j1 = 0; j1 < N; j1++) {
				for(i2 = i1; i2 < N; i2++) {
					for(j2 = j1; j2 < N; j2++) {
						sum = 0;//Initialize the
						// Calculate the sum of the submatrix
						for(m = i1; m <= i2; m++) {
							for(n = j1; n <= j2; n++)
								sum += *((int *)matrix + m * N + n);
						}
						//if a larger sum is found
						if(sum > max_sum) {
							max_sum = sum;//Update max_sum
							//Update submatrix indices
							start_row = i1;
							start_column = j1;
							end_row = i2;
							end_column = j2;
						}
					}
				}
			}
		}
	}

	// Call the print function to display the maximum submatrix
	print((int **)matrix, start_row, start_column, end_row, end_column, N, 1);

	stop = clock(); // Record the ending time of the algorithm
	printf("Algorithm 1 execution time: %.5fs\n\n", ((double)(stop - start)) / CLK_TCK); // Print the result

	return max_sum;
}

// Function to find the maximum matrix sum using the second algorithm
int max_matrix_sum2(int **matrix, int N) {
	start = clock(); // Record the starting time of the algorithm

	int sum, max_sum, start_row, start_column, end_row, end_column, i, j, m, n, p, column[N]; // Define the necessary variables
	int iteration;

	// Iterate the algorithm for K times
	for(iteration = 1; iteration <= K2; iteration++) {

		max_sum = 0;
		start_row = start_column = end_row = end_column = 0;

		// Nested loops to iterate through the matrix and find the maximum submatrix sum
		for(i = 0; i < N; i++) {
			// Initialize the column array
			for(m = 0; m < N; m++)
				column[m] = 0;
			for(j = i; j < N; j++) {
				// Update the column sums
				for(m = 0; m < N; m++)
					column[m] += *((int *)matrix + j * N + m);
				for(n = 0; n < N; n++) {
					sum = 0;
					for(p = n; p < N; p++) {
						// Calculate the sum of the submatrix using column sums
						sum += column[p];
						// if a larger sum is found
						if(sum > max_sum) {
							max_sum = sum;//Update max_sum
							//Update submatrix indices
							start_row = i;
							end_row = j;
							start_column = n;
							end_column = p;
						}
					}
				}
			}
		}
	}

	// Call the print function to display the maximum submatrix
	print((int **)matrix, start_row, start_column, end_row, end_column, N, 2);

	stop = clock(); // Record the ending time of the algorithm
	printf("Algorithm 2 execution time: %.5fs\n\n", ((double)(stop - start)) / CLK_TCK); // Print the result

	return max_sum;
}

// Function to find the maximum matrix sum using the third algorithm
int max_matrix_sum3(int **matrix, int N) {
	start = clock(); // Record the starting time of the algorithm

	int sum, max_sum, current_column, start_row, start_column, end_row, end_column, i, j, m, n, k, column[N]; // Define the necessary variables
	int iteration;

	// Iterate the algorithm for K times
	for(iteration = 1; iteration <= K3; iteration++) {

		max_sum = 0;
		current_column = start_row = start_column = end_row = end_column = 0;

		// Nested loops to iterate through the matrix and find the maximum submatrix sum
		for(i = 0; i < N; i++) {
			// Initialize the column array
			for(m = 0; m < N; m++)
				column[m] = 0;
			for(j = i; j < N; j++) {
				sum = 0, current_column = 0;
				for(k = 0; k < N; k++)
					column[k] += *((int *)matrix + j * N + k); // Update the column sums to zero
				for(n = 0; n < N; n++) {
					sum += column[n];
					// Update current_column and sum if sum becomes negative
					if(sum < 0) {
						sum = 0;
						current_column = n + 1;
					}
					//  if a larger sum is found
					if(sum > max_sum) {
						max_sum = sum;//Update max_sum
						//Update submatrix indices
						start_row = i;
						end_row = j;
						start_column = current_column;
						end_column = n;
					}
				}
			}
		}
	}

	// Call the print function to display the maximum submatrix
	print((int **)matrix, start_row, start_column, end_row, end_column, N, 3);

	stop = clock(); // Record the ending time of the algorithm
	printf("Algorithm 3 execution time: %.5fs\n\n", ((double)(stop - start)) / CLK_TCK); // Print the result

	return max_sum;
}

// Function to print the maximum submatrix
void print(int **matrix, int start_row, int start_column, int end_row, int end_column, int N, int x) {
	printf("The maximum submatrix obtained by Algorithm %d:\n", x);

	int m, n;
	//Nested loops to iterate through the matrix and print the maximum submatrix
	for(m = start_row; m <= end_row; m++) {
		for(n = start_column; n <= end_column; n++) {
			printf("%4d", *((int *)matrix + m * N + n));
		}
		printf("\n");
	}
}

