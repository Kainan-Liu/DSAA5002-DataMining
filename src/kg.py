from py2neo import Graph, Relationship, Node, Subgraph, NodeMatcher, RelationshipMatcher
from tqdm import tqdm
import os
import pandas as pd
import argparse

class Company_KG:
    def __init__(self, uri="bolt://localhost:7687", user = "neo4j", password = "lkn12345765") -> None:
        self.uri = uri
        self.user = user
        self.password = password

        self.graph = Graph(uri, auth=(user, password), name="neo4j")
        self.graph.delete_all()

        self.node_matcher = NodeMatcher(graph=self.graph)
        self.relation_matcher = RelationshipMatcher(graph=self.graph)

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
            password = input("please enter the Neo4j password: ")
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
    
    def implicit_mining(self, *, file_path = None, num_sample):
        print("Start mining the implicit company")
        assert file_path is not None, "file_path should not None"
        if os.path.exists(file_path):
            news = pd.read_excel(file_path).iloc[:num_sample, :]
            def get_positive_implicitCompany(value):
                name_list = [name.strip() for name in value.split(',')]
                positive_nodes = []
                for company_name in name_list:
                    node = self.node_matcher.match("Company", name=company_name).first()
                    relationships = list(self.relation_matcher.match(nodes=(node, None)))
                    for relationship in relationships:
                        if type(relationship).__name__ in ["cooperate", "invest", "same_industry", "supply"]:
                            positive_nodes.append(relationship.end_node["name"])
                if positive_nodes == []:
                    positive_nodes = None
                return positive_nodes
            
            def get_negative_implicitCompany(value):
                name_list = [name.strip() for name in value.split(',')]
                negative_nodes = []
                for company_name in name_list:
                    node = self.node_matcher.match("Company", name=company_name).first()
                    relationships = list(self.relation_matcher.match(nodes=(node, None)))
                    for relationship in relationships:
                        if type(relationship).__name__ in ["compete", "dispute"]:
                            negative_nodes.append(relationship.end_node["name"])
                if negative_nodes == []:
                    negative_nodes = None
                return negative_nodes

            tqdm.pandas()
            news["Implicit_Positive_Company"] = news["Explicit_Company"].progress_apply(get_positive_implicitCompany)
            news["Implicit_Negative_Company"] = news["Explicit_Company"].progress_apply(get_negative_implicitCompany)
            news.to_excel("./Data/Task2.xlsx", index=False)
        else:
            raise FileNotFoundError


def main(file_path, num_sample):
    assert file_path is not None, "file_path should not be None"
    graph_service = Company_KG()
    graph = graph_service.construct()
    print("✔ Success")
    print("===============================================")
    print("Implicit Company Mining")
    graph_service.implicit_mining(file_path=file_path, num_sample=num_sample)
    print("✔ Finish!")
    print("===============================================")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="KG")
    parser.add_argument("--file_path", default="./Data/Task1.xlsx")
    parser.add_argument("--num_sample", default=100, type=int)
    args = parser.parse_args()

    main(file_path=args.file_path, num_sample=args.num_sample)
