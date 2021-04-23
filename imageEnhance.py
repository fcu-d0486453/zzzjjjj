from misc.voc_xml_parser import VocParser




class ImEnhance:
    """

    """

    def __init__(self, xml_path):
        self.xml_path = VocParser(xml_path).get_dlist()

    def __len__(self):
        return len(self.xml_path)


    def __getitem__(self, idx):
