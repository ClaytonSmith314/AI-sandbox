from ExpertSystem.modules import Rule

axoims = [
    'Emily mother of Clayton',
    'Nathan father of Clayton',
    'Emily mother of Isaac',
    'Malisa mother of Emily',
    'Malisa wife of Tim',
    'Clayton brother of Isaac',
    'Isaac isboy',
    'Brigham isboy',
    'Clayton sibling of Brigham',
    'Eliza daughter of Emily',
    'Lydia isgirl',
    'Lydia sibling of Eliza',
    'Alison sister of Emily',
    'Mandy sister of Alison',
    'Jordan son of Alison',
    'Joey brother of Jordan',
    'Caitie sister of Jordan',
    'Evan son of Melanie',
    'Melanie child of Tim',
    'Vivian daughter of Melanie',
    'Clara sister of Evan',
    'Matthew brother of Vivian',
    'Malisa parent of Jenie',
    'Bryan brother of Timmy',
    'Bryan son of Jenie',
    'Mandy parent of Ellie',
    'Dave parent of Joey',
    'Ruben husband of Melanie',
    'Ellie sibling of Conner',
    'Ellie sibling of Zack',
    'Conner isboy',
    'Zack son of BigBryan',
    'Dave father of Will',
    'Bill father of Nathan',
    'Nathan brother of Sarah',
    'Sarah sister of Becca',
    'Becca sister of Luke',
    'Luke son of Katherine',
]

rules = (

    Rule([#1
        '$a _ $title of $b'
    ],
        '$a $title of $b'
    ),
    Rule([#2
        '$a $title of $b'
    ],
        '$a _ $title of $b'
    ),

    Rule([#3
        '$a $pfx father of $b'
    ],
        '$a $pfx parent of $b'
    ),
    Rule([#4
        '$a $pfx mother of $b'
    ],
        '$a $pfx parent of $b'
    ),

    Rule([#5
        '$b isboy',
        '$a $pfx parent of $b',
    ],
        '$b $pfx son of $a'
    ),
    Rule([#6
        '$a $pfx parent of $b',
        '$b isgirl'
    ],
        '$b $pfx daughter of $a'
    ),
    Rule([#7
        '$a $pfx parent of $b',
        '$a isboy'
    ],
        '$a $pfx father of $b'
    ),
    Rule([#8
        '$a $pfx parent of $b',
        '$a isgirl'
    ],
        '$a $pfx mother of $b'
    ),



    Rule([#9
        '$a $pfx father of $b'
    ],
        '$a isboy'
    ),
    Rule([#10
        '$a $pfx mother of $b'
    ],
        '$a isgirl'
    ),
    Rule([#11
        '$a $pfx son of $b'
    ],
        '$a isboy'
    ),
    Rule([#12
        '$a son of $b'
    ],
        '$a child of $b'
    ),
    Rule([#13
        '$a $pfx daughter of $b'
    ],
        '$a isgirl'
    ),
    Rule([#14
        '$a $pfx daughter of $b'
    ],
        '$a $pfx child of $b'
    ),
    Rule([#15
        '$a $pfx child of $b'
    ],
        '$b $pfx parent of $a'
    ),
    Rule([#16
        '$a wife of $b'
    ],
        '$a isgirl'
    ),
    Rule([#17
        '$a husband of $b'
    ],
        '$a isboy'
    ),



    Rule([#18
        '$a wife of $b'
    ],
        '$b husband of $a'
    ),
    Rule([#19
        '$a husband of $b'
    ],
        '$b wife of $a'
    ),


    Rule([#20
        '$a wife of $b',
        '$a $pfx mother of $c'
    ],
        '$b $pfx father of $c'
    ),
    Rule([#21
        '$a husband of $b',
        '$a $pfx father of $c'
    ],
        '$b $pfx mother of $c'
    ),
    Rule([#22
        '$a father of $c',
        '$b mother of $c'
    ],
        '$a husband of $b'
    ),


    Rule([#23
        '$a brother of $b'
    ],
        '$a isboy'
    ),
    Rule([#24
        '$a sister of $b'
    ],
        '$a isgirl'
    ),
    Rule([#25
        '$a brother of $b'
    ],
        '$a sibling of $b'
    ),
    Rule([#26
        '$a sister of $b'
    ],
        '$a sibling of $b'
    ),
    Rule([#27
        '$a sibling of $b'
    ],
        '$b sibling of $a'
    ),
    Rule([#28
        '$a sibling of $b',
        '$a isgirl'
    ],
        '$a sister of $b'
    ),
    Rule([#29
        '$a sibling of $b',
        '$a isboy'
    ],
        '$a brother of $b'
    ),
    Rule([#30
        '$a mother of $b',
        '$b parent of $c'
    ],
        '$a grand mother of $c'
    ),
    Rule([#31
        '$a father of $b',
        '$b parent of $c'
    ],
        '$a grand father of $c'
    ),

    Rule([#32
        '$a $pfx parent of $b'
    ],
        '$b $pfx child of $a'
    ),
    Rule([#33
        '$a mother of $b',
        '$b parent of $c'
    ],
        '$a grand mother of $c'
    ),


    Rule([#34
        '$a sibling of $b',
        '$a parent of $c',
        '$b parent of $d'
    ],
        '$c cousin of $d'
    ),

    Rule([#35
        '$a sibling of $b',
        '$b sibling of $c'
    ],
        '$a sibling of $c',
        unequal_variables=('$a','$b','$c')
    ),

    Rule([#36
        '$a parent of $b',
        '$a parent of $c'
    ],
        '$b sibling of $c',
        unequal_variables=('$a','$b','$c')
    ),
    Rule([#37
        '$a parent of $b',
        '$b sibling of $c'
    ],
        '$a parent of $c'
    ),

    Rule([#38
        '$a brother of $b',
        '$b parent of $c',
    ],
        '$a uncle of $c'
    ),
    Rule([#39
        '$a sister of $b',
        '$b parent of $c'
    ],
        '$a aunt of $c'
    ),
    Rule([#40
        '$a husband of $b',
        '$b aunt of $c'
    ],
        '$a uncle of $c'
    ),
    Rule([#41
        '$a wife of $b',
        '$b uncle of $c'
    ],
        '$a aunt of $c'
    ),
    Rule([#42
        '$a father of $c',
        '$b parent of $c'
    ],
        '$a husband of $b',
        unequal_variables=('$a', '$b', '$c')
    ),
    Rule([#43
        '$a mother of $c',
        '$b parent of $c'
    ],
        '$a wife of $b',
        unequal_variables=('$a', '$b', '$c')
    ),


)