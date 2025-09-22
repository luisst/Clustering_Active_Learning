import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from scipy import sparse
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class AdaptiveKNNGraphBuilder:
    """
    Build kNN graph with local scaling (Zelnik-Manor & Perona style) and mutual kNN filtering.
    Ensures connectivity via MST when needed.
    """
    
    def __init__(self, k: int = 10, sigma_method: str = 'adaptive', 
                 mutual_knn: bool = True, ensure_connected: bool = True,
                 verbose: bool = True):
        """
        Initialize the graph builder.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors
        sigma_method : str
            'adaptive' for local scaling, 'global' for single sigma
        mutual_knn : bool
            Whether to use mutual k-NN filtering
        ensure_connected : bool
            Whether to add MST edges to ensure connectivity
        verbose : bool
            Whether to print progress information
        """
        self.k = k
        self.sigma_method = sigma_method
        self.mutual_knn = mutual_knn
        self.ensure_connected = ensure_connected
        self.verbose = verbose
        
    def _compute_cosine_knn(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute k-NN using cosine distance."""
        if self.verbose:
            print(f"Computing {self.k}-NN with cosine distance...")
            
        # Use sklearn's NearestNeighbors with cosine metric
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric='cosine', n_jobs=-1)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Remove self-connections (first column)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        return distances, indices
    
    def _compute_local_sigma(self, distances: np.ndarray, method: str = 'kth') -> np.ndarray:
        """
        Compute local scaling parameter sigma_i for each point.
        
        Parameters:
        -----------
        distances : np.ndarray
            Distance matrix (n_samples, k)
        method : str
            'kth' - use k-th neighbor distance
            'median' - use median of k distances
            'mean' - use mean of k distances
        """
        if method == 'kth':
            # Use distance to k-th neighbor (Zelnik-Manor & Perona)
            sigma = distances[:, -1]  # Last column is k-th neighbor
        elif method == 'median':
            sigma = np.median(distances, axis=1)
        elif method == 'mean':
            sigma = np.mean(distances, axis=1)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-8)
        return sigma
    
    def _build_similarity_matrix(self, X: np.ndarray, distances: np.ndarray, 
                                indices: np.ndarray) -> sparse.csr_matrix:
        """Build similarity matrix with local scaling."""
        n_samples = X.shape[0]
        
        if self.sigma_method == 'adaptive':
            if self.verbose:
                print("Computing local scaling parameters...")
            sigma_i = self._compute_local_sigma(distances, method='kth')
        else:
            # Global sigma
            sigma_i = np.full(n_samples, np.median(distances))
        
        if self.verbose:
            print("Building similarity matrix with local scaling...")
            
        # Initialize sparse matrix
        row_indices = []
        col_indices = []
        similarities = []
        
        for i in range(n_samples):
            for j_idx, j in enumerate(indices[i]):
                if i != j:  # Skip self-loops
                    # Cosine distance to similarity conversion with local scaling
                    cos_dist = distances[i, j_idx]
                    
                    # Local scaling: exp(-d²/(sigma_i * sigma_j))
                    # For cosine distance, we use the distance directly
                    if self.sigma_method == 'adaptive':
                        sigma_ij = np.sqrt(sigma_i[i] * sigma_i[j])
                        similarity = np.exp(-cos_dist**2 / (sigma_ij**2 + 1e-8))
                    else:
                        similarity = np.exp(-cos_dist**2 / (sigma_i[i]**2 + 1e-8))
                    
                    row_indices.append(i)
                    col_indices.append(j)
                    similarities.append(similarity)
        
        # Create sparse matrix
        W = sparse.csr_matrix(
            (similarities, (row_indices, col_indices)),
            shape=(n_samples, n_samples)
        )
        
        return W
    
    def _apply_mutual_knn(self, W: sparse.csr_matrix) -> sparse.csr_matrix:
        """Apply mutual k-NN filtering."""
        if self.verbose:
            print("Applying mutual k-NN filtering...")
            
        # Convert to dense for easier manipulation (if memory allows)
        n_samples = W.shape[0]
        
        if n_samples < 10000:  # Use dense for smaller matrices
            W_dense = W.toarray()
            W_mutual = np.zeros_like(W_dense)
            
            # Keep edge only if both i->j and j->i exist
            for i in range(n_samples):
                for j in range(n_samples):
                    if W_dense[i, j] > 0 and W_dense[j, i] > 0:
                        # Take average of similarities
                        W_mutual[i, j] = (W_dense[i, j] + W_dense[j, i]) / 2
            
            W_mutual = sparse.csr_matrix(W_mutual)
        else:
            # Use sparse operations for larger matrices
            W_mutual = W.minimum(W.T)  # Element-wise minimum
            W_mutual = (W_mutual + W_mutual.T) / 2  # Symmetrize
        
        # Report filtering statistics
        original_edges = W.nnz
        mutual_edges = W_mutual.nnz
        if self.verbose:
            print(f"Edges before mutual filtering: {original_edges}")
            print(f"Edges after mutual filtering: {mutual_edges}")
            print(f"Filtering ratio: {mutual_edges/original_edges:.3f}")
        
        return W_mutual
    
    def _ensure_connectivity(self, W: sparse.csr_matrix, X: np.ndarray) -> sparse.csr_matrix:
        """Ensure graph connectivity by adding MST edges if needed."""
        if self.verbose:
            print("Checking graph connectivity...")
            
        # Check connectivity
        n_components, labels = connected_components(W, directed=False)
        
        if n_components == 1:
            if self.verbose:
                print("Graph is already connected.")
            return W
        
        if self.verbose:
            print(f"Graph has {n_components} components. Adding MST edges...")
        
        # Compute full distance matrix for MST
        # For large datasets, this might be memory intensive
        n_samples = X.shape[0]
        
        if n_samples < 5000:  # Use full distance matrix for smaller datasets
            full_distances = cosine_distances(X)
            # Convert to similarity for MST (use negative distances)
            mst = minimum_spanning_tree(-1.0 / (full_distances + 1e-8))
        else:
            # For larger datasets, use approximate MST via subsampling
            if self.verbose:
                print("Using approximate MST for large dataset...")
            # Sample representatives from each component
            representatives = []
            for comp in range(n_components):
                comp_indices = np.where(labels == comp)[0]
                # Take multiple representatives per component
                n_reps = min(10, len(comp_indices))
                reps = np.random.choice(comp_indices, n_reps, replace=False)
                representatives.extend(reps)
            
            representatives = np.array(representatives)
            sub_X = X[representatives]
            sub_distances = cosine_distances(sub_X)
            sub_mst = minimum_spanning_tree(-1.0 / (sub_distances + 1e-8))
            
            # Map back to original indices
            mst = sparse.lil_matrix((n_samples, n_samples))
            sub_mst_coo = sub_mst.tocoo()
            
            for i, j, data in zip(sub_mst_coo.row, sub_mst_coo.col, sub_mst_coo.data):
                orig_i, orig_j = representatives[i], representatives[j]
                # Use local scaling for MST edges too
                cos_dist = cosine_distances([X[orig_i]], [X[orig_j]])[0, 0]
                similarity = np.exp(-cos_dist**2 / 0.1)  # Fixed sigma for MST
                mst[orig_i, orig_j] = similarity
                mst[orig_j, orig_i] = similarity
            
            mst = mst.tocsr()
        
        if n_samples < 5000:
            # Convert MST similarities back and add to original graph
            mst_coo = mst.tocoo()
            W_lil = W.tolil()
            
            for i, j, data in zip(mst_coo.row, mst_coo.col, mst_coo.data):
                if W_lil[i, j] == 0:  # Only add if edge doesn't exist
                    cos_dist = cosine_distances([X[i]], [X[j]])[0, 0]
                    similarity = np.exp(-cos_dist**2 / 0.1)  # Fixed sigma for MST
                    W_lil[i, j] = similarity
                    W_lil[j, i] = similarity
            
            W = W_lil.tocsr()
        
        # Verify connectivity
        n_components_final, _ = connected_components(W, directed=False)
        if self.verbose:
            print(f"Final number of components: {n_components_final}")
        
        return W
    
    def build_graph(self, X: np.ndarray, 
                   msp_weights: Optional[np.ndarray] = None,
                   cluster_labels: Optional[np.ndarray] = None,
                   memberships: Optional[np.ndarray] = None) -> Tuple[sparse.csr_matrix, dict]:
        """
        Build the complete adaptive kNN graph.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature vectors (n_samples, 256)
        msp_weights : np.ndarray, optional
            MSP mutual reachability distances
        cluster_labels : np.ndarray, optional
            Predicted cluster labels
        memberships : np.ndarray, optional
            Normalized membership values (0 for outliers)
        
        Returns:
        --------
        W : sparse.csr_matrix
            Final similarity matrix
        info : dict
            Information about the graph construction process
        """
        if self.verbose:
            print("Starting adaptive kNN graph construction...")
            print(f"Data shape: {X.shape}")
            print(f"k: {self.k}, sigma_method: {self.sigma_method}")
            print(f"mutual_knn: {self.mutual_knn}, ensure_connected: {self.ensure_connected}")
        
        # Step 1: Compute k-NN with cosine distance
        distances, indices = self._compute_cosine_knn(X)
        
        # Step 2: Build similarity matrix with local scaling
        W = self._build_similarity_matrix(X, distances, indices)
        
        # Step 3: Apply mutual k-NN filtering if requested
        if self.mutual_knn:
            W = self._apply_mutual_knn(W)
        
        # Step 4: Ensure connectivity if requested
        if self.ensure_connected:
            W = self._ensure_connectivity(W, X)
        
        # Collect information
        info = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'k': self.k,
            'n_edges': W.nnz,
            'density': W.nnz / (X.shape[0] * (X.shape[0] - 1)),
            'sigma_method': self.sigma_method,
            'mutual_knn': self.mutual_knn,
            'ensure_connected': self.ensure_connected
        }
        
        if self.verbose:
            print("\nGraph construction completed!")
            print(f"Final graph: {info['n_samples']} nodes, {info['n_edges']} edges")
            print(f"Graph density: {info['density']:.6f}")
        
        return W, info
    
    def save_graph(self, W: sparse.csr_matrix, filename: str):
        """Save the graph to file."""
        sparse.save_npz(filename, W)
        if self.verbose:
            print(f"Graph saved to {filename}")
    
    @staticmethod
    def load_graph(filename: str) -> sparse.csr_matrix:
        """Load graph from file."""
        return sparse.load_npz(filename)


# Example usage and testing function
def create_sample_data(n_samples: int = 1000, n_features: int = 256, 
                      n_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Generate clustered data
    X = []
    cluster_labels = []
    
    for cluster in range(n_clusters):
        cluster_size = n_samples // n_clusters
        if cluster == n_clusters - 1:  # Last cluster gets remaining samples
            cluster_size = n_samples - (n_clusters - 1) * cluster_size
            
        # Random cluster center
        center = np.random.randn(n_features)
        center = center / np.linalg.norm(center)  # Normalize for cosine similarity
        
        # Generate samples around center
        cluster_data = np.random.randn(cluster_size, n_features) * 0.3
        cluster_data += center[np.newaxis, :]
        
        # Normalize each sample
        cluster_data = cluster_data / (np.linalg.norm(cluster_data, axis=1, keepdims=True) + 1e-8)
        
        X.append(cluster_data)
        cluster_labels.extend([cluster] * cluster_size)
    
    X = np.vstack(X)
    cluster_labels = np.array(cluster_labels)
    
    # Generate mock MSP weights (mutual reachability distances)
    msp_weights = np.random.exponential(1.0, n_samples)
    
    # Generate mock memberships (higher for non-outliers)
    memberships = np.random.beta(2, 1, n_samples)  # Skewed towards 1
    # Make some outliers
    n_outliers = int(0.05 * n_samples)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    memberships[outlier_indices] = 0.0  # Outliers have 0 membership
    
    return X, msp_weights, cluster_labels, memberships


def main():
    """Main function to demonstrate the graph builder."""
    print("=== Adaptive kNN Graph Builder Demo ===\n")
    
    # Create sample data
    X, msp_weights, cluster_labels, memberships = create_sample_data(n_samples=500, n_features=256)
    
    print("Sample data created:")
    print(f"- {X.shape[0]} samples with {X.shape[1]} features")
    print(f"- {len(np.unique(cluster_labels))} clusters")
    print(f"- {np.sum(memberships == 0)} outliers")
    print()
    
    # Test different configurations
    configs = [
        {"k": 10, "sigma_method": "adaptive", "mutual_knn": True, "ensure_connected": True},
        {"k": 15, "sigma_method": "adaptive", "mutual_knn": True, "ensure_connected": True},
        {"k": 10, "sigma_method": "global", "mutual_knn": False, "ensure_connected": True},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1} ---")
        print(f"Config: {config}")
        
        # Build graph
        builder = AdaptiveKNNGraphBuilder(**config, verbose=True)
        W, info = builder.build_graph(X, msp_weights, cluster_labels, memberships)
        
        # Analyze graph
        print(f"\nGraph analysis:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Check if graph is suitable for label propagation
        n_components, _ = connected_components(W, directed=False)
        print(f"  connectivity_check: {n_components} component(s)")
        
        # Save graph
        filename = f"knn_graph_config_{i+1}.npz"
        builder.save_graph(W, filename)
    
    print("\n=== Demo completed ===")


if __name__ == "__main__":
    main()