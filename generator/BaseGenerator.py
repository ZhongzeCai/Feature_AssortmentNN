class BaseGenerator:

    def __init__(

        self, 
        N_prod = 10, 

        Len_prodFeature = 3,
        Len_customerFeature = 5,

        **kwargs

    ):
        self.N_prod = N_prod
        self.Len_prodFeature = Len_prodFeature
        self.Len_customerFeature = Len_customerFeature

    def self_gen_instance(self):
        pass


    def prob_for_assortment(self, prod_assort, customer_feature):

        pass

    def gen_final_choice(self, prod_assort, customer_feature):

        pass


    def save_para(self, folder_path):
        pass



