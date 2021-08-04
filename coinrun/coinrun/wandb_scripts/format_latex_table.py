import numpy as np

table = r"""Env & \shortmethod (Ours) & PPO & PPO+DIAYN & PPO+CURL & DAAC & PPO+Sinkhorn & PPO+DBC\\ 
bigfish & \textbf{4.9$\pm$0.3} & 2.5$\pm$0.3 & 2.2$\pm$0.1 & 2.2$\pm$0.2 & 4.5$\pm$0.8 & 2.4$\pm$0.1 & 1.8$\pm$0.1 \\ 
bossfight & \textbf{8.6$\pm$0.2} & 6.2$\pm$0.5 & 1.1$\pm$0.2 & 5.4$\pm$0.9 & 1.9$\pm$1.2 & 6.1$\pm$0.5 & 5$\pm$0.1 \\ 
caveflyer & \textbf{4.8$\pm$0.2} & 4.7$\pm$0.4 & 1.0$\pm$1.9 & \textbf{4.8$\pm$0.3} & 4.3$\pm$0.0 & 4.7$\pm$0.1 & 3.6$\pm$0.1 \\ 
chaser & 7.3$\pm$0.2 & 7.5$\pm$0.2 & 5.6$\pm$0.5 & 7.4$\pm$0.1 & 7.3$\pm$0.1 & \textbf{7.6$\pm$0.2} & 4.8$\pm$0.1\\ 
climber & \textbf{6.2$\pm$0.2} & 5.8$\pm$0.1 & 0.8$\pm$0.7 & 5.7$\pm$0.4 & 5.8$\pm$0.4 & 5.7$\pm$0.4 & 4.1$\pm$0.4 \\ 
coinrun & \textbf{8.7$\pm$0.4} & 8.3$\pm$0.2 & 6.4$\pm$2.6 & 8.1$\pm$0.1 & 8.2$\pm$0.2 & 8.2$\pm$0.0 & 7.9$\pm$0.1\\ 
dodgeball & \textbf{1.9$\pm$0.1} & 1.3$\pm$0.1 & 1.4$\pm$0.2 & 1.5$\pm$0.1 & \textbf{1.9$\pm$0.0} & 1.6$\pm$0.1 & 1.0$\pm$0.3\\ 
fruitbot & \textbf{13.5$\pm$0.3} & 12.7$\pm$0.2 & 7.2$\pm$3.0 & \textbf{13.1$\pm$0.4} & 12.6$\pm$0.2 & 12.3$\pm$0.4 & 2$\pm$0.2\\ 
heist & 3.3$\pm$0.3 & 2.9$\pm$0.2 & 0.2$\pm$0.2 & 2.6$\pm$0.3 & \textbf{3.4$\pm$0.2} & 3.0$\pm$0.3 & 3.3$\pm$0.2 \\ 
jumper & 6.1$\pm$0.1 & 6.0$\pm$0.1 & 2.6$\pm$2.3 & 6.0$\pm$0.1 & \textbf{6.5$\pm$0.1} & 6.1$\pm$0.1 & 3.9$\pm$0.4\\ 
leaper & 2.6$\pm$0.3 & 3.3$\pm$0.9 & 2.5$\pm$0.2 & \textbf{3.8$\pm$0.9} & \textbf{3.8$\pm$0.6} & 3.2$\pm$0.8 & 2.7$\pm$0.1\\ 
maze & \textbf{5.7$\pm$0.1} & 5.5$\pm$0.0 & 1.6$\pm$1.2 & 5.5$\pm$0.1 & \textbf{5.7$\pm$0.3} & 5.5$\pm$0.3 & 5.0$\pm$0.1 \\ 
miner & 6.8$\pm$0.2 & 8.8$\pm$0.3 & 1.3$\pm$2.0 & \textbf{8.9$\pm$0.1} & 6.3$\pm$0.6 & 8.8$\pm$0.5 & 4.8$\pm$0.1\\ 
ninja & \textbf{5.8$\pm$0.1} & 5.4$\pm$0.2 & 2.8$\pm$2.0 & 5.7$\pm$0.2 & 5.5$\pm$0.2 & 5.4$\pm$0.3 & 3.5$\pm$0.1\\ 
plunder & \textbf{6.7$\pm$0.4} & 6.2$\pm$0.4 & 2.1$\pm$2.5 & \textbf{6.7$\pm$0.2} & 4.0$\pm$0.2 & 6.0$\pm$0.9 & 5.1$\pm$0.1\\ 
starpilot & \textbf{7.9$\pm$0.5} & 5.2$\pm$0.2 & 5.8$\pm$0.6 & 5.2$\pm$0.2 & 4.7$\pm$0.4 & 5.2$\pm$0.2 & 2.8$\pm$0.1\\
"""

prefix = r"""\begin{table}[ht]
\centering
\caption{Average eval returns collected after 8M of training frames, $\pm$ one standard deviation.} 
\resizebox{\linewidth}{!}{%
\begin{tabular}{l||l|llllll}
\toprule
"""

suffix = r"""\bottomrule
\end{tabular}%
}
\label{tab:procgen}
\end{table}%
"""

table_new = []
for row in table.split('\n'):
    row_new = []
    for el in row.split('&'):
        row_new.append(el.strip().rstrip('\\'))
    if len(row_new) > 1:
        table_new.append(row_new)
table_new = np.array(table_new)

def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]

# table_new = np.delete(table_new,3,1)
# swap_cols(table_new,4,6)
# swap_cols(table_new,3,5)
table_new = table_new[:,[0,2,5,7,3,6,4,1]]

main_table_str = ""
for i,el in enumerate(table_new[0]):
    if i < len(table_new[0])-1:
        main_table_str += el + " & "
    else:
        main_table_str += el + ' \\n'
        
print(main_table_str)

for i,row in enumerate(table_new[1:]):
    row_acc = row[0]
    for j,col in enumerate(row[1:]):
        row_acc += " & " + col

    print(row_acc+r' \\')