from collections import defaultdict
import numpy as np
import os, logging, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from scipy.linalg import solve
import json
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class SkillManager:
    def __init__(
        self,
        retrieval_top_k=8,
        basic_tools = [],
        ckpt_dir="ckpt",
        resume=False,
    ):  
        os.makedirs(f"{ckpt_dir}/skill/code", exist_ok=True)
        os.makedirs(f"{ckpt_dir}/skill/description", exist_ok=True)
        os.makedirs(f"{ckpt_dir}/skill/vectordb", exist_ok=True)
        matrix_path = f"{ckpt_dir}/skill/skills_matrix.npy"
        if os.path.exists(matrix_path):
            self.skills_matrix = np.load(matrix_path) 
        else:
            self.skills_matrix = np.full((200, 200), np.inf)
            np.save(matrix_path, self.skills_matrix)

        self.basic_tools = basic_tools
      
        if resume:
            logging.info(f"\033[33mLoading Skill Manager from {ckpt_dir}/skill\033[0m")
            with open(f"{ckpt_dir}/skill/skills.json", 'r') as f:
                self.skills = json.load(f)
        else:
            self.skills = {}
        self.layer_to_skill = defaultdict(list)
        for skill_name in self.skills.keys():
            self.layer_to_skill[self.skills[skill_name]["level"]].append(skill_name)
        self.retrieval_top_k = retrieval_top_k
        self.ckpt_dir = ckpt_dir
        
        self.vectordb = Chroma(
            collection_name="skill_vectordb",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"{ckpt_dir}/skill/vectordb",
        )
        
        assert self.vectordb._collection.count() == len(self.skills), (
            f"Skill Manager's vectordb is not synced with skills.json.\n"
            f"There are {self.vectordb._collection.count()} skills in vectordb but {len(self.skills)} skills in skills.json.\n"
            f"Did you set resume=False when initializing the manager?\n"
            f"You may need to manually delete the vectordb directory for running from scratch."
        )

    @property
    def programs(self):
        programs = ""
        for skill_name, entry in self.skills.items():
            programs += f"{entry['code']}\n\n"
        for primitives in self.basic_tools:
            programs += f"{primitives}\n\n"
        return programs

    def add_new_skill(self, info, **kwargs):
        duplicate_code = kwargs.pop('duplicate_tool', [])
        neighbor_indices = kwargs.pop('neighbor_indices', [])

        program_name = info["name"]
        program_code = info["code"]
        program_docstring = info["docstring"]
        program_demo = info["demo"]
    
        logging.info(
            f"\033[33mSkill Manager generated description for {program_name}:\n{program_docstring}\033[0m"
        )
        id2skill = {}
        for skill in self.skills.keys():
            id2skill[self.skills[skill]["id"]] = self.skills[skill]   

        for tool in duplicate_code:
            if tool in self.skills.keys():
                find_duplicate = False
                duplicate_code_id = self.skills[tool]["id"]
                
                for idx, value in enumerate(self.skills_matrix[duplicate_code_id, :]):
                    if not np.isinf(value):
                        if idx in id2skill and id2skill[idx]["level"] > id2skill[duplicate_code_id]["level"]:
                            find_duplicate = True
                            break
                if not find_duplicate:
                    duplicate_code_name = id2skill[duplicate_code_id]["name"]
                    self.vectordb._collection.delete(ids=[duplicate_code_name])
                    self.skills.pop(duplicate_code_name, None)
                    self.skills_matrix[:, duplicate_code_id] = np.inf
                    self.skills_matrix[duplicate_code_id, :] = np.inf
                    logging.info(f"Replace duplicated tool {duplicate_code_name}")

        if program_name in self.skills.keys():
            find = False
            if program_code == self.skills[program_name]["code"]:
                logging.info(f"{program_name} is directly used to solve the task.")
                return
            else:
                i = 2
                while f"{program_name}V{i}.py" in os.listdir(f"{self.ckpt_dir}/skill/code"):
                    i += 1
                dumped_program_name = f"{program_name}V{i}"
            '''
            program_id = self.skills[program_name]["id"] 
            for idx, value in enumerate(self.skills_matrix[program_id, :]):
                if not np.isinf(value):
                    if idx in id2skill and id2skill[idx]["level"] > self.skills[program_name]["level"]:
                        find = True
                        break
            if not find:
                self.vectordb._collection.delete(ids=[program_name])
                self.skills.pop(program_name, None)
                self.skills_matrix[:, program_id] = np.inf
                self.skills_matrix[program_id, :] = np.inf
                dumped_program_name = program_name
                logging.info(f"\033[33mSkill {program_name} already exists. Rewriting!\033[0m")
            '''
        else:
            dumped_program_name = program_name
            
        id = max([self.skills[skill_name]["id"] for skill_name in self.skills.keys()]) + 1 if self.skills else 0
        skills = neighbor_indices
        for skill, _ in skills:
            if skill in self.skills:
                sub_tool_id = self.skills[skill]["id"]
                self.skills_matrix[sub_tool_id][id] = 1.0
                self.skills_matrix[id][sub_tool_id] = 1.0
                logging.info(f"Score between {skill} and {dumped_program_name} is 1.0")        

        self.vectordb.add_texts(
            texts=[program_docstring],
            ids=[dumped_program_name],
            metadatas=[{"name": dumped_program_name, "level": info["level"], "id": id}],
        )

        self.skills[dumped_program_name] = {
            "name": dumped_program_name,
            "code": program_code,
            "docstring": program_docstring,
            "id": id,
            "level": info["level"],
            "freq": 0,
            "demo": program_demo
        }
        
        self.layer_to_skill = defaultdict(list)
        for skill_name in self.skills.keys():
            self.layer_to_skill[self.skills[skill_name]["level"]].append(skill_name)

        assert self.vectordb._collection.count() == len(
            self.skills.keys()
        ), "vectordb is not synced with skills.json"
        
        with open(f"{self.ckpt_dir}/skill/code/{dumped_program_name}.py", 'w') as f:
            f.write(program_code)
        with open(f"{self.ckpt_dir}/skill/skills.json", 'w') as f:
            json.dump(self.skills, f, indent=4)
        np.save(f"{self.ckpt_dir}/skill/skills_matrix.npy", self.skills_matrix)
        
    def retrieve_skills(self, query, damping_factor=0.4):
        """
        Retrieve skills using PageRank algorithm.
        :param query: The query to search skills for.
        :param damping_factor: The damping factor for PageRank (default 0.85).
        :param tol: Convergence threshold for PageRank (default 1e-6).
        :param max_iter: Maximum iterations for PageRank (default 100).
        :return: A list of top-k skill dictionaries with names and code.
        """
        k = min(self.vectordb._collection.count(), self.retrieval_top_k)
        if k == 0:
            return []
        
        desc_and_scores = self.vectordb.similarity_search_with_score(query, k=len(self.skills))
        
        similarity_scores = defaultdict(list)
        for skill, score in desc_and_scores:
            skill_name = skill.metadata.get("name")
            similarity_scores[skill_name].append(1 - score)

        similarity_scores = {
            skill_name: sum(scores) / len(scores)
            for skill_name, scores in similarity_scores.items()
        }

        # Retrieve top-k skills based on similarity scores
        top_k_skill_names = [
            skill for skill, _ in sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)[:k]
        ]
        logging.info("\033[33mSimilarity-based Top-k Skills:\033[0m")
        for skill_name in top_k_skill_names:
            logging.info(f"Skill: {skill_name}, Similarity Score: {similarity_scores[skill_name]:.4f}")

        id_to_skill = {}
        for skill in self.skills.keys():
            id_to_skill[self.skills[skill]["id"]] = skill


        prior_weights = {skill: similarity_scores[skill] for skill in self.skills.keys()}
        total_prior = sum(prior_weights.values())
        normalized_prior_weights = {skill: weight / total_prior for skill, weight in prior_weights.items()}
        prior_sorted_skills = sorted(normalized_prior_weights.items(), key=lambda x: x[1], reverse=True)[: k+3]
        logging.info("\033[32mOriginal Prior-based Top-k Skills:\033[0m")
        for skill, weight in prior_sorted_skills:
            logging.info(f"Skill: {skill}, Prior Score: {weight:.8f}")
            
        id_mapping = {skill_data["id"]: idx for idx, (skill, skill_data) in enumerate(self.skills.items())}
        mapped_skill_count = len(id_mapping)
    
        transition_matrix = np.zeros((mapped_skill_count, mapped_skill_count))

        for skill, skill_data in self.skills.items():
            original_skill_id = skill_data["id"]
            if original_skill_id not in id_mapping:
                continue
            mapped_skill_id = id_mapping[original_skill_id] 

            for neighbor_idx, weight in enumerate(self.skills_matrix[original_skill_id, :]):
                if not np.isinf(weight):
                    neighbor = id_to_skill.get(neighbor_idx, None)
    
                    if neighbor is not None and self.skills[neighbor]["id"] in id_mapping:
                        mapped_neighbor_idx = id_mapping[self.skills[neighbor]["id"]]
                        transition_matrix[mapped_skill_id, mapped_neighbor_idx] = 1.0
        
        for j in range(mapped_skill_count):
            col_sum = transition_matrix[:, j].sum()
            if col_sum > 0:
                transition_matrix[:, j] /= col_sum 
            else:
                transition_matrix[:, j] = 1 / mapped_skill_count 
                
        ranks = np.array([normalized_prior_weights[skill] for skill in self.skills.keys() if self.skills[skill]["id"] in id_mapping])
        N = len(ranks)
        I = np.eye(N)
        M = I - damping_factor * transition_matrix.T
        b = (1 - damping_factor) * ranks
        new_ranks = solve(M, b)

        scores = {skill: new_ranks[id_mapping[skill_data["id"]]] for skill, skill_data in self.skills.items() if skill_data["id"] in id_mapping}
        top_k_skill_names = [skill for skill, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)][:k]
        top_k_skill_names = list(
            sorted(
                top_k_skill_names,
                key=lambda item: (self.skills[item].get("freq", 0), scores[item]),
                reverse=True,
            )
        )
        pagerank_sorted_skills = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k + 3]
        logging.info("\033[33mPageRank-based Top-k Skills:\033[0m")
        for skill, score in pagerank_sorted_skills:
            logging.info(f"Skill: {skill}, PageRank Score: {score:.8f}")
        logging.info(
            f"\033[33mSkill Manager retrieved skills: "
            f"{', '.join([skill_name for skill_name in top_k_skill_names])}\033[0m"
        )
        skills = []
        for skill_name in top_k_skill_names:
            skills.append({"name": skill_name, "code": self.skills[skill_name]["code"], "docstring": self.skills[skill_name]["docstring"], "demo": self.skills[skill_name]["demo"]
                           , "level": self.skills[skill_name]["level"]})
        return skills
      
    def pruning(self, T_0):
        import math
        T_0 = math.log10(T_0)
        id_to_skill = {}
        
        logging.info("====Begin Pruning====")
        for skill in self.skills.keys():
            id_to_skill[self.skills[skill]["id"]] = skill
        max_level = max(list(self.layer_to_skill.keys()))
        parent_pruned = {}

        cannot_prune = set()
        for level in range(max_level, 0, -1):
            T_l = T_0 / (1 + 0.8 * math.log(level, 2))
            for skill in self.layer_to_skill[level]:
                if skill in cannot_prune:
                    parent_pruned[skill] = False
                    continue

                if self.skills[skill]["freq"] >= T_l:
                    parent_pruned[skill] = False
                    continue

                node_weight = self.skills_matrix[self.skills[skill]["id"], :]
                neighbors_idx = np.where(~np.isinf(node_weight))[0]
                cutting = True
                for idx in neighbors_idx:
                    neighbor = id_to_skill[idx]
                    if self.skills[neighbor]["level"] <= level:
                        continue
                    if not parent_pruned.get(neighbor, True): 
                        cutting = False
                        break

                if cutting:
                    logging.info(
                        f"Pruning skill: {skill} (Level: {level}, Frequency: {self.skills[skill]['freq']:.2f}, Threshold: {T_l:.2f})"
                    )
                else:
                    for idx in neighbors_idx:
                        neighbor = id_to_skill[idx]
                        if self.skills[neighbor]["level"] < level:
                            cannot_prune.add(neighbor)

                parent_pruned[skill] = cutting

        for skill in parent_pruned.keys():
            if parent_pruned[skill]:
                id = self.skills[skill]["id"]
                self.skills_matrix[id, :] = np.inf
                self.skills_matrix[:, id] = np.inf
                self.skills.pop(skill, "")
                self.vectordb._collection.delete(ids=[skill])

        with open(f"{self.ckpt_dir}/skill/skills.json", 'w') as f:
            json.dump(self.skills, f, indent=4)
