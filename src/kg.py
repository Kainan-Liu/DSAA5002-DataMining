from py2neo import Graph, Relationship, Node, Subgraph, NodeMatcher
import os
import pandas as pd


class Company_KG:
    def __init__(self, uri="bolt://localhost:7687", user = "neo4j", password = "lkn12345765") -> None:
        self.uri = uri
        self.user = user
        self.password = password

        self.graph = Graph(uri, auth=(user, password), name="neo4j")
        self.graph.delete_all()

        self.node_matcher = NodeMatcher(graph=self.graph)

    def _create_CompanyNode(self, *, file_path=None, company_name=None, company_code=None, companny_id=None):
        node_list = []
        if file_path is not None:
            if os.path.exists(file_path):
                company = pd.read_csv(file_path)

                for i in range(len(company)):
                    node_list.append(Node("Company",
                                          name=company.loc[i, "company_name"],
                                          code=(company.loc[i, "code"]),
                                          id=int(company.loc[i, ":ID"])))
            else:
                raise FileNotFoundError("File_path not exists")
        elif company_name or company_code or companny_id:
            node_list.append(Node("Company", name=company_name, code=company_code, id=companny_id))
        else:
            print("Please provide file containes company information or explicitly first")
        
        subgraph = Subgraph(node_list)
        tx = self.graph.begin()
        tx.create(subgraph=subgraph)
        self.graph.commit(tx=tx)

        return node_list
    
    def _create_CompanyRelation(self, *, node_list: list, file_path=None, node1=None, node2=None, relation=None):
        relation_list = []
        assert node_list is not None, "please provide the node_list[Node]"
        if file_path is not None:
            if os.path.exists(file_path):
                relation_table = pd.read_csv(file_path)
                for i in range(len(relation_table)):
                    node1 = self.node_matcher.match("Company", id=int(relation_table.loc[i, ":START_ID"])).first()
                    relation = relation_table.loc[0, ":TYPE"]
                    node2 = self.node_matcher.match("Company", id=int(relation_table.loc[i, ":END_ID"])).first()
                    
                    relation_list.append(Relationship(node1, relation, node2))
            else:
                raise FileNotFoundError("File_path not exists")
        elif node1 or node2:
            relation_list.append(Relationship(node1, relation, node2))
        else:
            print("Please provide file containes company relationship information or provide explicitly first")

        tx = self.graph.begin()
        subgraph = Subgraph(nodes=node_list, relationships=relation_list)
        tx.create(subgraph=subgraph)

        self.graph.commit(tx=tx)

    
    def construct(self):
        fail_time = 0
        while fail_time < 5:
            password = input("please enter the Neo4j key: ")
            if password != "lkn12345765":
                fail_time += 1
                print("====================================")
                print("Invalid Password, please enter again")
                print("====================================")
            else:
                break
        if fail_time + 1 == 5:
            raise ValueError("Invalid Password")
        
        print("===============================================")
        print("==> Welcome")
        print("Start Constructing Knowledge Graph!")
        node_list = self._create_CompanyNode(file_path="./KnowledgeGraph/hidy.nodes.company.csv")
        self._create_CompanyRelation(node_list=node_list, file_path="./KnowledgeGraph/hidy.relationships.compete.csv")
        self._create_CompanyRelation(node_list=node_list, file_path="./KnowledgeGraph/hidy.relationships.cooperate.csv")
        self._create_CompanyRelation(node_list=node_list, file_path="./KnowledgeGraph/hidy.relationships.dispute.csv")
        self._create_CompanyRelation(node_list=node_list, file_path="./KnowledgeGraph/hidy.relationships.invest.csv")
        self._create_CompanyRelation(node_list=node_list, file_path="./KnowledgeGraph/hidy.relationships.same_industry.csv")
        self._create_CompanyRelation(node_list=node_list, file_path="./KnowledgeGraph/hidy.relationships.supply.csv")
        
        return self.graph

if __name__ == "__main__":
    graph_service = Company_KG()
    graph = graph_service.construct()
    print("✔ Success")
    print("===============================================")
