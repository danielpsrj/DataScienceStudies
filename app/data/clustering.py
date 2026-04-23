"""
Clustering-specific data for the Clustering concept page.
"""

from typing import List, Dict


def get_clustering_applications() -> List[Dict[str, str]]:
    """Get real-world applications for clustering analysis."""
    return [
        {
            "title": "Customer Segmentation",
            "description": "Group customers based on purchasing behavior, demographics, and preferences.",
            "details": "Enables targeted marketing, personalized recommendations, and improved customer retention.",
            "examples": [
                "E-commerce customer segmentation",
                "Banking customer profiling",
                "Subscription service user groups",
            ],
        },
        {
            "title": "Image Segmentation",
            "description": "Group pixels in images based on color, texture, or other features.",
            "details": "Used in computer vision for object detection, medical imaging, and autonomous vehicles.",
            "examples": [
                "Medical image analysis (tumor detection)",
                "Satellite image classification",
                "Facial recognition systems",
            ],
        },
        {
            "title": "Anomaly Detection",
            "description": "Identify unusual patterns or outliers in data.",
            "details": "Critical for fraud detection, network security, and system monitoring.",
            "examples": [
                "Credit card fraud detection",
                "Network intrusion detection",
                "Manufacturing defect detection",
            ],
        },
        {
            "title": "Document Clustering",
            "description": "Group similar documents for organization and retrieval.",
            "details": "Used in information retrieval, recommendation systems, and content management.",
            "examples": [
                "News article categorization",
                "Research paper organization",
                "Customer support ticket grouping",
            ],
        },
    ]


def get_clustering_pitfalls() -> List[Dict[str, str]]:
    """Get common pitfalls for clustering analysis."""
    return [
        {
            "title": "Choosing Wrong Number of Clusters",
            "description": "Selecting too many or too few clusters, leading to over-segmentation or under-segmentation.",
            "severity": "High",
            "solution": [
                "Use elbow method to find optimal k",
                "Apply silhouette analysis",
                "Use gap statistic method",
                "Consider domain knowledge and business requirements",
            ],
            "tips": "There's no one-size-fits-all answer. Different methods may suggest different optimal k values.",
        },
        {
            "title": "Ignoring Feature Scaling",
            "description": "Using unscaled features when distance-based algorithms (like K-means) are sensitive to scale.",
            "severity": "High",
            "solution": [
                "Always scale features before clustering",
                "Use standardization (z-score normalization)",
                "Consider min-max scaling for bounded ranges",
                "Use algorithms less sensitive to scale (DBSCAN with appropriate parameters)",
            ],
            "tips": "Distance-based algorithms treat all dimensions equally. A feature with larger range will dominate the distance calculation.",
        },
        {
            "title": "Assuming Clusters are Spherical",
            "description": "Using algorithms that assume spherical clusters (like K-means) for non-spherical data.",
            "severity": "Medium",
            "solution": [
                "Visualize data in 2D/3D to understand cluster shapes",
                "Use density-based clustering (DBSCAN) for arbitrary shapes",
                "Consider hierarchical clustering",
                "Use spectral clustering for complex structures",
            ],
            "tips": "K-means works well for spherical, equally sized clusters. For other shapes, explore different algorithms.",
        },
        {
            "title": "Interpreting Clusters Without Validation",
            "description": "Assuming clusters have meaningful interpretation without proper validation.",
            "severity": "Medium",
            "solution": [
                "Validate clusters with domain experts",
                "Use internal validation metrics (silhouette score, Davies-Bouldin index)",
                "Compare with ground truth if available",
                "Test stability with different initializations",
            ],
            "tips": "Clusters are mathematical constructs, not necessarily meaningful business segments. Always validate with domain knowledge.",
        },
        {
            "title": "Ignoring High-Dimensionality Issues",
            "description": "Applying clustering directly to high-dimensional data without dimensionality reduction.",
            "severity": "Medium",
            "solution": [
                "Apply PCA or t-SNE for dimensionality reduction",
                "Use feature selection to reduce dimensions",
                "Consider subspace clustering methods",
                "Use algorithms designed for high dimensions",
            ],
            "tips": "In high dimensions, distance measures become less meaningful (curse of dimensionality). Dimensionality reduction is often essential.",
        },
    ]


def get_clustering_references() -> List[Dict[str, str]]:
    """Get references for clustering analysis."""
    return [
        {
            "type": "paper",
            "authors": "MacQueen, J.",
            "year": "1967",
            "title": "Some methods for classification and analysis of multivariate observations",
            "journal": "Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability",
            "url": "https://projecteuclid.org/proceedings/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Fifth-Berkeley-Symposium-on-Mathematical-Statistics-and/Chapter/Some-methods-for-classification-and-analysis-of-multivariate-observations/bsmsp/1200512992",
            "abstract": "The k-means algorithm is one of the most popular clustering algorithms. This paper introduces the original k-means algorithm.",
            "tags": ["k-means", "foundational", "clustering"],
        },
        {
            "type": "paper",
            "authors": "Ester, M., Kriegel, H.-P., Sander, J., & Xu, X.",
            "year": "1996",
            "title": "A density-based algorithm for discovering clusters in large spatial databases with noise",
            "journal": "Proceedings of the Second International Conference on Knowledge Discovery and Data Mining",
            "url": "https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf",
            "abstract": "DBSCAN, a density based clustering algorithm, is presented. It can discover clusters of arbitrary shape and is robust to noise.",
            "tags": ["DBSCAN", "density-based", "noise-resistant"],
        },
        {
            "type": "book",
            "authors": "Kaufman, L., & Rousseeuw, P. J.",
            "year": "1990",
            "title": "Finding Groups in Data: An Introduction to Cluster Analysis",
            "publisher": "Wiley",
            "doi": "10.1002/9780470316801",
            "abstract": "This book presents a comprehensive introduction to cluster analysis. It covers various clustering methods and provides practical guidance on their application.",
            "tags": ["cluster analysis", "comprehensive", "methods"],
        },
        {
            "type": "paper",
            "authors": "Rousseeuw, P. J.",
            "year": "1987",
            "title": "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis",
            "journal": "Journal of Computational and Applied Mathematics",
            "url": "https://www.sciencedirect.com/science/article/pii/0377042787901257",
            "doi": "10.1016/0377-0427(87)90125-7",
            "abstract": "A graphical display is proposed for partitioning techniques, where each cluster is represented by a so-called silhouette, which is based on the comparison of its tightness and separation.",
            "tags": ["silhouette", "validation", "visualization"],
        },
        {
            "type": "online",
            "authors": "Scikit-learn Developers",
            "year": "2023",
            "title": "Clustering - scikit-learn documentation",
            "publisher": "scikit-learn",
            "url": "https://scikit-learn.org/stable/modules/clustering.html",
            "abstract": "Documentation for clustering algorithms in scikit-learn, including k-means, DBSCAN, hierarchical clustering, and more.",
            "tags": ["python", "scikit-learn", "documentation"],
        },
    ]