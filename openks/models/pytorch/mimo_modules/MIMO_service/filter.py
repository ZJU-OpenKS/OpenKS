from nltk import pos_tag

def tuple_filter(query):

    def get_subject(tpl):
        return get_subject_concept(tpl), get_subject_attribute(tpl)

    def get_object(tpl):
        return get_object_concept(tpl), get_object_attribute(tpl)

    def get_object_concept(tpl):
        return tpl[3]

    def get_subject_concept(tpl):
        return tpl[0]

    def get_object_attribute(tpl):
        return tpl[4]

    def get_subject_attribute(tpl):
        return tpl[1]

    def get_relation(tpl):
        return tpl[2]


    def missing_relation(stmt):
        conditions = stmt['condition tuples']
        facts = stmt['fact tuples']
        for con_tpl in conditions:
            if get_relation(con_tpl) == 'NIL':
                stmt['condition tuples'].remove(con_tpl)
        for fac_tpl in facts:
            if get_relation(fac_tpl) == 'NIL':
                stmt['fact tuples'].remove(fac_tpl)
        return stmt

    def redundant_fact_conflict_with_condition(stmt):
        conditions = stmt['condition tuples']
        facts = stmt['fact tuples']
        for con_tpl in conditions:
            con_sub = get_subject_concept(con_tpl) # B -> C
            con_obj = get_object_concept(con_tpl)
            if con_obj == 'NIL' or con_sub == 'NIL':
                continue
            fac_tpl1, fac_tpl2 = None, None
            for tpl in facts:
                if con_sub == get_object_concept(tpl):
                    fac_tpl1 = tpl # A -> B
                elif con_obj == get_object_concept(tpl):
                    fac_tpl2 = tpl # A -> C

            if not fac_tpl1 or not fac_tpl2:
                continue

            if get_subject(fac_tpl1) == get_subject(fac_tpl2):
                stmt['fact tuples'].remove(fac_tpl2)

        return stmt


    def missing_concept(stmt):
        conditions = stmt['condition tuples']
        facts = stmt['fact tuples']
        for con_tpl in conditions:
            if get_object(con_tpl) == ('NIL', 'NIL')\
                and get_subject(con_tpl) == ('NIL', 'NIL'):
                stmt['condition tuples'].remove(con_tpl)
        for fac_tpl in facts:
            if get_object(fac_tpl) == ('NIL', 'NIL') \
                and get_subject(fac_tpl) == ('NIL', 'NIL'):
                stmt['fact tuples'].remove(fac_tpl)
        return stmt

    def redundant_condition_circle(stmt):

        conditions = stmt['condition tuples']
        for con_tpl in conditions:
            con_sub = get_subject_concept(con_tpl) # B -> C
            con_obj = get_object_concept(con_tpl)
            if con_obj == 'NIL' or con_sub == 'NIL':
                continue
            con_tpl1, con_tpl2 = None, None
            for tpl in conditions:
                if con_sub == get_object_concept(tpl) and tpl != con_tpl:
                    con_tpl1 = tpl # A -> B
                elif con_obj == get_object_concept(tpl) and tpl != con_tpl:
                    con_tpl2 = tpl # A -> C

            if not con_tpl1 or not con_tpl2:
                continue

            if get_subject(con_tpl1) == get_subject(con_tpl2):
                relation = get_relation(con_tpl2).split()
                pos = pos_tag(relation)
                if pos[0][1] == 'IN':
                    stmt['condition tuples'].remove(con_tpl2)

        return stmt

    query = missing_concept(query)
    query = missing_relation(query)
    query = redundant_condition_circle(query)
    query = redundant_fact_conflict_with_condition(query)

    return query



# a = {'text': 'Here , we demonstrate that reprogramming with lentiviruses carrying the iPSC-inducing factors ( Oct4-Sox2-Klf4-cMyc , OSKM ) caused senescence in mouse fibroblasts , establishing a stress barrier for cell reprogramming .',
#  'fact tuples': [['NIL', 'NIL', 'demonstrate that', 'NIL', 'NIL'],
#                  ['reprogramming', 'NIL', 'caused', 'mouse fibroblasts', 'NIL'],
#                  ['reprogramming', 'NIL', 'caused', 'senescence', 'NIL']],
#  'condition tuples': [['reprogramming', 'NIL', 'with', 'iPSC-inducing factors', 'NIL'],
#                       ['reprogramming', 'NIL', 'with', 'lentiviruses', 'NIL'],
#                       ['lentiviruses', 'NIL', 'in', 'iPSC-inducing factors', 'NIL'],
#                       ['lentiviruses', 'NIL', 'NIL', 'Oct4-Sox2-Klf4-cMyc', 'NIL'],
#                       ['senescence', 'NIL', 'in', 'mouse fibroblasts', 'NIL']],
#  'unit_indx': [[5, 20, 21, 5, 18, 28, 29, 25, 26, 5, 15, 5, 10, 11, 5, 7, 7, 15, 7, 13, 7, 10, 11, 5, 10, 11, 18, 20, 21, 28, 29],
#                [], [3, 4, 17, 17, 23, 23, 6, 6, 6, 8, 8, 8, 8, 19, 27]]}


# print(tuple_filter(a))