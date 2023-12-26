import random
## SIMULATION
def generate_dummy_knowledge_graph(file_path, num_triples=10):
    with open(file_path, 'w') as file:
        for _ in range(num_triples):
            subject = "<node_{0}>".format(random.randint(1, 5))
            predicate = "<relationship_{0}>".format(random.randint(1, 5))
            object_ = "<node_{0}>".format(random.randint(1, 5))
            
            triple = "{0} {1} {2} .\n".format(subject, predicate, object_)
            file.write(triple)

if __name__ == "__main__":
    generate_dummy_knowledge_graph("dummy_graph.nt", num_triples=10)
