# pylint: skip-file

import os
from assimilator import *
from Boinc import boinc_project_path

class SlimeClustersAssimilator(Assimilator):
	def __init__(self):
		Assimilator.__init__(self)

	def assimilate_handler(self, wu, results, canonical_result):
		if canonical_result == None:
			return
		
		src_file = self.get_file_path(canonical_result)
		dst_dir = boinc_project_path.project_path('slime-clusters')
		dst_file = os.path.join(dst_dir, 'results.txt')

		if not os.path.exists(dst_dir):
			os.makedirs(dst_dir)
		
		with open(src_file, 'r') as src, open(dst_file, 'a') as dst:
			dst.writelines(src.readlines())

if __name__ == "__main__":
	SlimeClustersAssimilator().run()