#include <iostream>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>

using namespace std;
// Heavily referencing https://github.com/rexdwyer/MPI-K-means-clustering/blob/master/kmeans.c

// Stop checking when the centroids move less than the threshold
#define THRESHOLD 0.00001

// Distance between two individual sites
double euclidean_distance(double *, double *, int);

// Assigns a site to its proper cluster
// Calculate the distance to each cluster from the site
// Takes the cluster with the lowest distance
int assign_site(double *, double *, int, int);

// Adds a site to a vector of all sites
void add_site(double *, double *, int);

// Creates random sites
// All values are from 0-1
double * rand_sites(int);

void print_centroids(double * centroids, const int cluster_num, const int dimension_num);
int main(int argc, char **argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc != 4 && rank == 0) {
		printf("Usage is as follows: \nmpirun -n 100 ./kmeans elements_per_cluster cluster_num dimension_num");
		return 0;
	}

	int elements_per = atoi(argv[1]);
	int cluster_num = atoi(argv[2]);
	int dimension_num = atoi(argv[3]);

	// Seed the random generator
	srand(time(NULL));

	// Declaring buffers for both individual process and the process handler - process 0
	// First will be for process handler
	// Second for individual processes
	// Matrixes are declared as a single array b/c the dimensionality is variable


	// All sites possible
	// Matrix but accessed and parsed like an array
	double * all_sites = NULL;
	double * sites = sites = new double[elements_per * dimension_num];

	// Sum of site distances assigned to each cluster
	// Matrix but accessed and parsed like an array
	double * all_sums = NULL;
	double * sums = new double[cluster_num * dimension_num];

	// Number of sites assigned to a cluster
	int * all_site_count = NULL;
	int * site_count = new int[cluster_num];

	// Cluster assignments for each site
	int * all_assignments = NULL;
	int * cluster_assignment = new int[elements_per];

	// All centroids. Will be broadcasted later
	// Matrix but accessed and parsed like an array
	double * centroids = new double[cluster_num * dimension_num];

	if (rank == 0) {
		// Size is number of processes
		all_sites = rand_sites(elements_per * dimension_num * size);
		// initial seeding for clusters is the first every other site (k times)
		// hopefully a little more random than just the first k sites
		for (int i = 0; i < cluster_num * dimension_num; i++) {
			centroids[i] = all_sites[i];
		}
		print_centroids(centroids, cluster_num, dimension_num);
		all_site_count = new int[cluster_num];
		all_sums = new double[cluster_num * dimension_num];
		all_assignments = new int[elements_per * size];
	}

	// Scatter all_sites to each process's sites
	MPI_Scatter(all_sites, elements_per * dimension_num, MPI_DOUBLE, sites, elements_per * dimension_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Measures the distance the new centroids are from the old centroids
	double norm = 1.0;
	double * site = sites;
	while (norm > THRESHOLD) {
		// Broadcast all centroids to processes to compare and adjust
    MPI_Bcast(centroids, cluster_num * dimension_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		// Reset all sum and site counts to 0
		for (int i = 0; i < cluster_num; i++) {
			site_count[i] = 0;
		}
		for (int i = 0; i < cluster_num * dimension_num;i++) {
			sums[i] = 0;
		}


		// Increment site by dimension to get the next site
		for (int i = 0; i < elements_per; i++, site += dimension_num) {
			int cluster = assign_site(site, centroids, cluster_num, dimension_num);
			site_count[cluster] += 1;
			add_site(site, &sums[cluster * dimension_num], dimension_num);
		}

		MPI_Reduce(sums, all_sums, cluster_num * dimension_num, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(site_count, all_site_count, cluster_num, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		// Compute the new centroids by dividing sums
		// Compute how much the centroids have moved
		if (rank == 0) {
			cout << endl;
			for (int i = 0; i < cluster_num; i++) {
				for (int j = 0; j < dimension_num; j++) {
					all_sums[dimension_num * i + j] /= all_site_count[i];
					cout << all_sums[dimension_num * i + j] << " ";
				}
				cout << endl;
			}

			// Calculate how much the centroids have changed
			// all_sums now holds the location for new centroids
			// centroids holds the location for old centroids
			// Compare the distance between the two to see if they've moved
			norm = euclidean_distance(all_sums, centroids, cluster_num * dimension_num);

			// New centroids are in all_sums
			// Copy all coordinates from all_sums to centroids
			for (int i = 0; i < cluster_num * dimension_num; i++) {
				centroids[i] = all_sums[i];
        cout << centroids[i] << endl;
			}
		}
		// Broadcast the new locations of centroids
		MPI_Bcast(&norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	// New centroids have been calculated
	// Assign labels - Tell each site which centroid they belong to
	// Done locally in each process
	site = sites;
	for (int i = 0; i < elements_per; i++, site += dimension_num) {
		cluster_assignment[i] = assign_site(site, centroids, cluster_num, dimension_num);
	}

	// Gather all labels into the master process - p0
	MPI_Gather(cluster_assignment, elements_per, MPI_INT, all_assignments, elements_per, MPI_INT, 0, MPI_COMM_WORLD);
	// Root can print out all sites and labels.
  if (rank == 0) {
  	double * site = all_sites;
    print_centroids(centroids, cluster_num, dimension_num);
    for (int i = 0; i < size * elements_per; i++, site += dimension_num) {
    	cout << "site: ";
    	for (int j = 0; j < dimension_num; j++) cout << site[j] << ", ";
    	cout <<"\n\tlabel: " << all_assignments[i] << "\n";
    }
  }
	free(sites);
	free(sums);
	free(cluster_assignment);
	free(site_count);
	free(centroids);
	if (rank == 0) {
		free(all_sites);
		free(all_sums);
		free(all_assignments);
		free(all_site_count);
	}
	MPI_Finalize();
	return 0;
}

double euclidean_distance(double * site1, double * site2, int dimension) {
	double distance = 0;
	double difference = 0;
	for (int i = 0; i < dimension; i++) {
		difference = site1[i] - site2[i];
		distance += difference * difference;
	}
	return distance;
}

int assign_site(double * site, double * centroids, int cluster_num, int dimension) {
	int assigned_cluster = 0;
	// Start lowest distance on the first cluster
	double low_distance = euclidean_distance(site, centroids, dimension);
	// Increment by dimension to get the next centroid
	double * current_centroid = centroids + dimension;

	for (int i = 1; i < cluster_num; i++, current_centroid += dimension) {
		double distance = euclidean_distance(site, current_centroid, dimension);
		if (distance < low_distance) {
			assigned_cluster = i;
			low_distance = distance;
		}
	}
	return assigned_cluster;
}

void add_site(double * site, double * sums, int dimension) {
	for (int i = 0; i < dimension; i++) {
		sums[i] += site[i];
	}
}

double * rand_sites(int total_elements) {
	double * all_elements = new double[total_elements];
	double range = 5;
	for (int i = 0; i < total_elements; i++) {
		// rand_num is 0-1
		double rand_num = (double)rand() / (double) RAND_MAX;
		// all elements are now between 0-5
		all_elements[i] = rand_num * range;
	}
	return all_elements;
}

void print_centroids(double * centroids, const int cluster_num, const int dimension_num) {
  double *p = centroids;
  cout << "Centroids:\n";
  for (int i = 0; i< cluster_num; i++) {
    for (int j = 0; j< dimension_num; j++, p++) {
      cout << *p << ", ";
    }
    cout << "\n";
  }
}
