/*
Viet Le
Simon Lee
Abdulrahman Alharbi

CPSC 479
Professor Doina Bein
Spring 2019

Semester Project - Implementing K-Means Clustering with MPI in C++
*/

#include <iostream>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

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
// All values are from 0-5
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
  double * local_sites = new double[elements_per * dimension_num];

  // Holds the sums of the centroids. Divide them by all_site_count[i] to get means for centroids
  double * all_sums = NULL;
  double * local_sums = new double[cluster_num * dimension_num];

  // Holds the number of sites a cluster has
  int * all_site_count = NULL;
  int * local_site_count = new int[cluster_num];

  // Holds the cluster a site is assigned to
  int * all_labels = NULL;
  int * local_labels = new int[elements_per];

  // Holds the coordinates for the centroids
  double * centroids = new double[cluster_num * dimension_num];

  // Process 0 randomly generates data
  if (rank == 0) {
		// Size is number of processes
		all_sites = rand_sites(elements_per * dimension_num * size);
		// Initial seeding for centroids are the first cluster_num centroids
		for (int i = 0; i < cluster_num * dimension_num; i++) {
			centroids[i] = all_sites[i];
		}
    cout << "Initial Centroids:" << endl;
		print_centroids(centroids, cluster_num, dimension_num);
		all_site_count = new int[cluster_num];
		all_sums = new double[cluster_num * dimension_num];
		all_labels = new int[elements_per * size];
	}
  // Distribute the sites to its local process
	MPI_Scatter(all_sites, elements_per * dimension_num, MPI_DOUBLE, local_sites, elements_per * dimension_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // Measures how much the centroids have moved
	// If the centroids has moved less than the threshold, stop the loop - that's good enough
  double has_moved = 1.0;
  double * site = local_sites;
  while (has_moved > THRESHOLD) {
    // Broadcast all centroids to processes to compare and adjust
    MPI_Bcast(centroids, cluster_num * dimension_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Reset all sum and site counts to 0
    for (int i = 0; i < cluster_num; i++) {
      local_site_count[i] = 0;
    }
    for (int i = 0; i < cluster_num * dimension_num;i++) {
      local_sums[i] = 0.0;
    }
		// Reset the pointer to the beginning of the local array
		// Without this you'll move out of bounds and get nan values for the centroids
		site = local_sites;
    // Don't need to reset all_site_count and all_sums because we reduce them later
    // One iteration for each site in local_sites
    // Iterate site by dimension_num to get start of site coordinates
    for (int i = 0; i < elements_per; i++, site += dimension_num) {
      int cluster = assign_site(site, centroids, cluster_num, dimension_num);
      local_site_count[cluster] += 1;
      // Add site's coordinates to local_sums
      add_site(site, &local_sums[cluster * dimension_num], dimension_num);
    }

    MPI_Reduce(local_sums, all_sums, cluster_num * dimension_num, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(local_site_count, all_site_count, cluster_num, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // all_sums now contains ALL sites added together
    // to get new centroid coordinates, divide all_sums by all_site_count
    if (rank == 0) {
      int index;
      for (int i = 0; i < cluster_num; i++) {
        for (int j = 0; j < dimension_num; j++) {
          index = dimension_num * i + j;
          all_sums[index] /= all_site_count[i];
        }
      }
      // all_sums now contains the coordinates for new centroids
      // centroids holds locations for old centroids
      // check how much each individual centroid has moved
      // only exit the loop when centroids have moved less than THRESHOLD
      has_moved = euclidean_distance(all_sums, centroids, cluster_num * dimension_num);

      // Copy new centroid coordinates into centroids
      for (int i = 0; i < cluster_num * dimension_num; i++) {
        centroids[i] = all_sums[i];
      }
    }
    // Broadcast the new has_moved
		MPI_Bcast(&has_moved, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  // Final centroids have been assigned
  // Tell each site which cluster they belong to
  site = local_sites;
  for (int i = 0; i < elements_per; i++, site += dimension_num) {
    local_labels[i] = assign_site(site, centroids, cluster_num, dimension_num);
  }
  MPI_Gather(local_labels, elements_per, MPI_INT, all_labels, elements_per, MPI_INT, 0, MPI_COMM_WORLD);

  // Process 0 now has everything. Finish by printing
  if (rank == 0) {
    double * site = all_sites;
    cout << "Final Centroids: " << endl;
    print_centroids(centroids, cluster_num, dimension_num);
    for (int i = 0; i < size * elements_per; i++, site += dimension_num) {
        cout << "Site: ";
        for (int j = 0; j < dimension_num; j++) cout << site[j] << ", ";
        cout <<"\n\tCluster: " << all_labels[i] << "\n";
    }
  }

  free(local_sites);
  free(local_sums);
  free(local_site_count);
  free(local_labels);
  free(centroids);
  if (rank == 0) {
    free(all_sites);
    free(all_sums);
    free(all_site_count);
    free(all_labels);
  }

  MPI_Finalize();
  return 0;
}

double euclidean_distance(double * site1, double * site2, int dimension) {
	double distance = 0;
	double difference = 0;
	for (int i = 0; i < dimension; i++) {
		difference = site1[i] - site2[i];
		distance += pow(difference, 2);
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
  double * centroid = centroids;
  for (int i = 0; i< cluster_num; i++) {
    for (int j = 0; j< dimension_num; j++, centroid++) {
      cout << * centroid << ", ";
    }
    cout << endl;
  }
	cout << endl;
}
