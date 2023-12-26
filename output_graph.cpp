#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

struct Node {
    int id;
    std::string label;
};

struct Relationship {
    int id;
    std::string label;
};

struct Edge {
    int sourceId;
    int targetId;
    double weight;
};

std::map<int, Node> readNodeMappings(const std::string& nodeIDFile) {
    std::map<int, Node> nodeMappings;
    std::ifstream file(nodeIDFile);
    if (file.is_open()) {
        int nodeId;
        std::string nodeLabel;
        while (file >> nodeLabel >> nodeId) {
            Node node = {nodeId, nodeLabel};
            nodeMappings[nodeId] = node;
            std::cout << "Read Node: " << nodeLabel << " ID: " << nodeId << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Error: Unable to open Node ID file." << std::endl;
    }
    return nodeMappings;
}

std::map<int, Relationship> readRelationshipMappings(const std::string& edgeIDFile) {
    std::map<int, Relationship> relationshipMappings;
    std::ifstream file(edgeIDFile);
    if (file.is_open()) {
        int relationshipId;
        std::string relationshipLabel;
        while (file >> relationshipLabel >> relationshipId) {
            Relationship relationship = {relationshipId, relationshipLabel};
            relationshipMappings[relationshipId] = relationship;
            std::cout << "Read Relationship: " << relationshipLabel << " ID: " << relationshipId << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Error: Unable to open Relationship ID file." << std::endl;
    }
    return relationshipMappings;
}

std::vector<double> readWeights(const std::string& weightsFile) {
    std::vector<double> weights;
    std::ifstream file(weightsFile);
    if (file.is_open()) {
        double weight;
        while (file >> weight) {
            weights.push_back(weight);
            std::cout << "Read Weight: " << weight << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Error: Unable to open Weights file." << std::endl;
    }
    return weights;
}
void constructInputGraph(const std::string& nodeIDFile,
                         const std::string& edgeIDFile, const std::string& weightsFile, const std::string& outputFile) {
    std::cout << "Constructing Input Graph..." << std::endl;

    std::map<int, Node> nodeMappings = readNodeMappings(nodeIDFile);
    std::map<int, Relationship> relationshipMappings = readRelationshipMappings(edgeIDFile);
    std::vector<double> weights = readWeights(weightsFile);

    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open output file." << std::endl;
        return;
    }

    std::ifstream file(edgeIDFile);
    if (file.is_open()) {
        int entityId1, entityId2, relationshipId;
        while (file >> entityId1 >> entityId2 >> relationshipId) {
            // Get node and relationship labels using mappings
            Node sourceNode = nodeMappings[entityId1];
            Relationship edgeLabel = relationshipMappings[relationshipId];
            Node targetNode = nodeMappings[entityId2];

            // Create edge with weight
            Edge edge = {sourceNode.id, targetNode.id, weights[relationshipId]};
            std::cout << "Constructed Edge: <" << sourceNode.label << "> <" << edgeLabel.label << "> <" << targetNode.label << ">." << std::endl;

            // Write the edge to the output file
            outFile << "<" << sourceNode.label << "> <" << edgeLabel.label << "> <" << targetNode.label << "> . " << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Error: Unable to open Edge ID file." << std::endl;
    }

    outFile.close();
    std::cout << "Input Graph construction complete. Output written to " << outputFile << std::endl;
}

int main() {
    std::string nodeIDFile = "knowledge_graph.ntnode_ID";
    std::string edgeIDFile = "knowledge_graph.ntedge_ID";
    std::string weightsFile = "here.txt";
    std::string outputFile = "output_graph.nt";

    constructInputGraph(nodeIDFile, edgeIDFile, weightsFile, outputFile);

    return 0;
}
