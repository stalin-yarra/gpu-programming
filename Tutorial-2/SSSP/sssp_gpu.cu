#include <iostream>
#include <vector>
#include <fstream>
#include <climits>
#include <cuda_runtime.h>

using namespace std;

#define INF INT_MAX

// CUDA error-checking helper
void checkCuda(cudaError_t error, const char *message)
{
    if (error != cudaSuccess)
    {
        cerr << "CUDA Error at " << message
             << ": " << cudaGetErrorString(error)
             << endl;
        exit(1);
    }
}

// --------------------------------------------------
// GPU kernel
// --------------------------------------------------
__global__ void relaxKernel(
    int V,
    const int *rowPtr,
    const int *col,
    const int *value,
    int *dist,
    int *changed)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= V)
        return;

    if (dist[u] == INF)
        return;

    int start = rowPtr[u];
    int end = rowPtr[u + 1];

    for (int i = start; i < end; i++)
    {
        int v = col[i];
        int w = value[i];

        int newDist = dist[u] + w;

        int oldDist = atomicMin(&dist[v], newDist);

        if (newDist < oldDist)
        {
            atomicExch(changed, 1);
        }
    }
}

int main()
{
    // --------------------------------------------------
    // 1. Read graph
    // --------------------------------------------------

    ifstream file("graph.txt");

    if (!file)
    {
        cerr << "Error: Could not open graph.txt" << endl;
        return 1;
    }

    int V, E;

    file >> V >> E;

    cout << "Vertices: " << V << endl;
    cout << "Edges: " << E << endl;

    vector<int> src(E);
    vector<int> dest(E);
    vector<int> weight(E);

    for (int i = 0; i < E; i++)
    {
        file >> src[i] >> dest[i] >> weight[i];
    }

    file.close();

    // --------------------------------------------------
    // 2. Build CSR
    // --------------------------------------------------

    vector<int> rowPtr(V + 1, 0);
    vector<int> col(E);
    vector<int> value(E);

    for (int i = 0; i < E; i++)
    {
        rowPtr[src[i] + 1]++;
    }

    for (int i = 1; i <= V; i++)
    {
        rowPtr[i] += rowPtr[i - 1];
    }

    vector<int> position = rowPtr;

    for (int i = 0; i < E; i++)
    {
        int u = src[i];

        int index = position[u];

        col[index] = dest[i];
        value[index] = weight[i];

        position[u]++;
    }

    cout << "\nCSR Representation\n";

    cout << "rowPtr: ";
    for (int x : rowPtr)
        cout << x << " ";
    cout << endl;

    cout << "col:    ";
    for (int x : col)
        cout << x << " ";
    cout << endl;

    cout << "value:  ";
    for (int x : value)
        cout << x << " ";
    cout << endl;


    // --------------------------------------------------
    // 3. Allocate GPU memory
    // --------------------------------------------------

    int *d_rowPtr;
    int *d_col;
    int *d_value;
    int *d_dist;
    int *d_changed;

    checkCuda(
        cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int)),
        "cudaMalloc d_rowPtr"
    );

    checkCuda(
        cudaMalloc(&d_col, E * sizeof(int)),
        "cudaMalloc d_col"
    );

    checkCuda(
        cudaMalloc(&d_value, E * sizeof(int)),
        "cudaMalloc d_value"
    );

    checkCuda(
        cudaMalloc(&d_dist, V * sizeof(int)),
        "cudaMalloc d_dist"
    );

    checkCuda(
        cudaMalloc(&d_changed, sizeof(int)),
        "cudaMalloc d_changed"
    );


    // --------------------------------------------------
    // 4. Copy CSR Host → Device
    // --------------------------------------------------

    checkCuda(
        cudaMemcpy(
            d_rowPtr,
            rowPtr.data(),
            (V + 1) * sizeof(int),
            cudaMemcpyHostToDevice
        ),
        "copy rowPtr"
    );

    checkCuda(
        cudaMemcpy(
            d_col,
            col.data(),
            E * sizeof(int),
            cudaMemcpyHostToDevice
        ),
        "copy col"
    );

    checkCuda(
        cudaMemcpy(
            d_value,
            value.data(),
            E * sizeof(int),
            cudaMemcpyHostToDevice
        ),
        "copy value"
    );


    // --------------------------------------------------
    // 5. Initialize distances
    // --------------------------------------------------

    vector<int> initialDist(V, INF);

    int source = 0;

    initialDist[source] = 0;

    checkCuda(
        cudaMemcpy(
            d_dist,
            initialDist.data(),
            V * sizeof(int),
            cudaMemcpyHostToDevice
        ),
        "copy initial distances"
    );


    // --------------------------------------------------
    // 6. Launch GPU SSSP
    // --------------------------------------------------

    int threadsPerBlock = 256;

    int blocks =
        (V + threadsPerBlock - 1)
        / threadsPerBlock;

    cout << "\nLaunching GPU kernel..." << endl;

    int changed = 1;

    int iteration = 0;

    while (changed)
    {
        iteration++;

        int zero = 0;

        checkCuda(
            cudaMemcpy(
                d_changed,
                &zero,
                sizeof(int),
                cudaMemcpyHostToDevice
            ),
            "reset changed"
        );

        relaxKernel<<<blocks, threadsPerBlock>>>(
            V,
            d_rowPtr,
            d_col,
            d_value,
            d_dist,
            d_changed
        );

        checkCuda(
            cudaGetLastError(),
            "kernel launch"
        );

        checkCuda(
            cudaDeviceSynchronize(),
            "kernel execution"
        );

        checkCuda(
            cudaMemcpy(
                &changed,
                d_changed,
                sizeof(int),
                cudaMemcpyDeviceToHost
            ),
            "copy changed"
        );

        cout << "Iteration "
             << iteration
             << " completed."
             << endl;
    }


    // --------------------------------------------------
    // 7. Copy result Device → Host
    // --------------------------------------------------

    vector<int> dist(V);

    checkCuda(
        cudaMemcpy(
            dist.data(),
            d_dist,
            V * sizeof(int),
            cudaMemcpyDeviceToHost
        ),
        "copy final distances"
    );


    // --------------------------------------------------
    // 8. Print result
    // --------------------------------------------------

    cout << "\nGPU SSSP Result\n";

    for (int v = 0; v < V; v++)
    {
        cout << "Vertex "
             << v
             << " : ";

        if (dist[v] == INF)
            cout << "INF";
        else
            cout << dist[v];

        cout << endl;
    }


    // --------------------------------------------------
    // 9. Free GPU memory
    // --------------------------------------------------

    cudaFree(d_rowPtr);
    cudaFree(d_col);
    cudaFree(d_value);
    cudaFree(d_dist);
    cudaFree(d_changed);

    return 0;
}
