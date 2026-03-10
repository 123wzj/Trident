先阅读docs目录下的文档，里面有你之前做的工作你了解一下，再到build_dataset目录下写一个脚本，统计pulse_ids.json文件里面拥有pulse数量最多的20个组织，然后结合pull.py文件拉取这20个组织的pulse，把不在pulse_ids.json文件里面的pluse放到新的json文件作为后续增量更新知识图谱以及增量训练的数据。把前后工作连通起来


拉取的新组织确保其别名的文件也被拉取不要被遗漏，拉取后的数据有被处理过吗能让
1. 使用 incremental_update.py 更新知识图谱
2. 使用 neo4j2pytorch_incremental.py 导出增量特征
3. 使用 incremental_train_fusion.py 进行增量训练
这几步去使用吗？