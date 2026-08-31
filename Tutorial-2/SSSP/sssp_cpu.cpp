#include <iostream>
#include <vector>
#include <fstream>
#include <climits>

using namespace std;

#define INF INT_MAX

int main()
{
    ifstream file("graph.txt");

    if (!file)
    {
        cout << "Error: Could not open graph.txt" << endl;
        return 1;
    }

    int V, E;
    file >> V >> E;

    vector<int> src(E);
    vector<int> dest(E);
    vector<int> weight(E);

    for (int i = 0; i < E; i++)
    {
        file >> src[i] >> dest[i] >> weight[i];
    }

    file.close();

    // Build CSR
    vector<int> rowPtr(V + 1, 0);
    vector<int> col(E);
    vector<int> value(E);

    for (int i = 0; i < E; i++)
        rowPtr[src[i] + 1]++;

    for (int i = 1; i <= V; i++)
        rowPtr[i] += rowPtr[i - 1];

    vector<int> position = rowPtr;

    for (int i = 0; i < E; i++)
    {
        int u = src[i];
        int index = position[u];

        col[index] = dest[i];
        value[index] = weight[i];

        position[u]++;
    }

    cout << "CSR Representation\n";

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

    // CPU Dijkstra
    int source = 0;

    vector<int> dist(V, INF);
    vector<bool> visited(V, false);

    dist[source] = 0;

    for (int count = 0; count < V; count++)
    {
        int u = -1;
        int minDist = INF;

        for (int v = 0; v < V; v++)
        {
            if (!visited[v] && dist[v] < minDist)
            {
                minDist = dist[v];
                u = v;
            }
        }

        if (u == -1)
            break;

        visited[u] = true;

        for (int i = rowPtr[u]; i < rowPtr[u + 1]; i++)
        {
            int v = col[i];
            int w = value[i];

            if (dist[u] != INF &&
                dist[v] > dist[u] + w)
            {
                dist[v] = dist[u] + w;
            }
        }
    }

    cout << "\nCPU SSSP Result\n";

    for (int v = 0; v < V; v++)
    {
        cout << "Vertex " << v << " : ";

        if (dist[v] == INF)
            cout << "INF";
        else
            cout << dist[v];

        cout << endl;
    }

    return 0;
}
