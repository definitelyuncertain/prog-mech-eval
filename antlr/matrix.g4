grammar matrix;

parse
 : statements? EOF
 ;

statements
 : expr ( ';' expr )* ( ';' )*
 ;

expr:
    function                                 # FunctionExpr
    | atom = NUMBER                              # NumberExpr
    | '(' expr ')'                             # ParenExpr
    | atom = VAR                                 # VarExpr
    ;

function
 : fnname=ID '(' args=arguments? ')'           # FunctionCallExpr
 ;

arguments
 : expr ( ',' expr )*
 ;

VAR: ('A') ;
NUMBER: ( [0-9]* '.' )? [0-9]+;
ID: [a-zA-Z_] [a-zA-Z0-9_]* ;
WS   : [ \t]+ -> skip ;
NEWLINE   : [ \n]+ -> skip ;