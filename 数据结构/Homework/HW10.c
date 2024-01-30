#include <stdlib.h>

#define MAX_N 501000
#define MAX_M 109999
#define IFN 1e9 

typedef struct {
    int v1, v2;
    long int w;
} Edge;

int subsets[MAX_N];
Edge t[MAX_M];
int kruskal(Edge edges[], int n, int m) ;

int cmp(const void *a, const void *b) {
    return ((Edge *)a)->w - ((Edge *)b)->w;
}

int find(int subsets[], int i) {
    if (subsets[i] != i)
        subsets[i] = find(subsets, subsets[i]);
    return subsets[i];
}

void Union(int subsets[], int x, int y) {
    subsets[find(subsets, y)] = x;
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    Edge edges[m];
    
    for (int i = 0; i < m; i++) {
        scanf("%d %d %d", &edges[i].v1, &edges[i].v2, &edges[i].w);
    }
	
    int min = kruskal(edges, n, m), min2;

    if (min != -1) {
        printf("%d\n", min);

        Edge edgesCopy[m];
        
		for (int i = 0; i < n - 1; i++) {
            for(int j = 0; j < m; j++){
            	edgesCopy[j] = t[j];
			}
			int temp = edgesCopy[i].w;
			edgesCopy[i].w = IFN;
			min2 = kruskal(edgesCopy, n, m);
			if (min == min2) {
            	printf("NO\n");
            	return 0;
        	}
			edgesCopy[i].w = temp;
        }
        printf("Yes\n");
        
    } else {
        printf("No MST\n");
    }

    return 0;
}

int kruskal(Edge edges[], int n, int m) {
    qsort(edges, m, sizeof(Edge), cmp);
    
    for (int i = 1; i <= n; i++)
        subsets[i] = i;
    
    long int Weight = 0;
    int Count = 0;

    for (int i = 0; i < m; i++) {
        int v1 = edges[i].v1;
        int v2 = edges[i].v2;

        if (find(subsets, v1) != find(subsets, v2)) {
            Union(subsets, v1, v2);
            Weight += edges[i].w;
            t[Count++] = edges[i]; 
        }

        if (Count == n - 1) break;
    }
	
	if(Count == n - 1)
		return Weight;
    else 
		return -1;
}

