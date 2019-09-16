from dataset.vcdb2 import VCDB



if __name__ == '__main__':
    db=VCDB()

    q=db.query_list[0]
    print(q)
    l=[[(l[0]['start'],l[0]['end']),l[1]] for l in db.get_relative_pair(q) if not(q['end']< l[0]['start'] or q['start']>l[0]['end']) and (q['name']!=l[1]['name'])]
    print(l)

