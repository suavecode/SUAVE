# save.py
#
# Created: T. Lukaczyk Feb 2015
# Updated:  

""" SUAVE Methods for IO """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Data import Data as XML_Data
from SUAVE.Core import Data

from warnings import warn

# Try to load a suitable celementtree incarnation. This is supposed to work for
# both Python 2.4 (both python and C version) and Python 2.5
try:
	import xml.etree.ElementTree as et # python 2.5
except ImportError:
	try:
		import cElementTree as et # python 2.4 celementtree
	except ImportError:
		import elementtree.ElementTree as et # python 2.4 elementtree
try:
	et
except NameError:
	warn('XML.save.py: No suitable XML package found. Install python-elementtree (default in python 2.5).',ImportWarning)


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def save(xml_data,filename):

	# translate to xml data if not already
	if not isinstance(xml_data,XML_Data):
		xml_data = XML_Data.from_dict(xml_data)


	# element builder
	def to_element(prev,data):

		# new node with tag
		if prev is None:
			node = et.Element(data.tag)
		else:
			node = et.SubElement(prev, data.tag)

		# attributes
		for k,v in data.attributes.items():
			node.set(k,v)

		# content
		if data.content:
			node.text = data.content

		# elements
		for elem in data.elements:
			# recursion!
			to_element(node,elem)

		#new line


		return node

	# run the builder
	xml_data = to_element(None,xml_data)

	# apply indentation
	indent(xml_data)

	# to xml tree
	tree = et.ElementTree(xml_data)

	# write!
	output = open(filename,'w')
	tree.write(output, 'utf-8')
	output.write('\n')    
	output.close()

	return


def indent(elem, level=0):
	# Indentation helper from http://effbot.org/zone/element-lib.htm
	i = "\n" + level*"  "
	if len(elem):
		if not elem.text or not elem.text.strip():
			elem.text = i + "  "
		for elem in elem:
			indent(elem, level+1)
		if not elem.tail or not elem.tail.strip():
			elem.tail = i
	else:
		if level and (not elem.tail or not elem.tail.strip()):
			elem.tail = i