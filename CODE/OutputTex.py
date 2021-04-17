# 3. Function to output latex regression table
def out_latex(file, tex_dict):

    ############################################################################
    # Input Notes
    ############################################################################

    string = """
    \\begin{center} $(\\mathrm{%s}, N=%s, T=%s)$ \\
        \\begin{tabular}{cccccccc}
            \\hline \\hline 
            & & PC & PLS & \\multicolumn{2}{c}{Ridge} & \\multicolumn{2}{c}{LF} \\\\
            \\hline 
            & $r$ & $k$ & $k$ & $\\alpha$ & DOF & $\\alpha$ & DOF \\\\
            \\hline 
            DGP 1 & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\
            (s.e.) & $-$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ \\\\
            DGP 2 & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\
            (s.e.) & $-$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ \\\\
            DGP 3 & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\
            (s.e.) & $-$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ \\\\
            DGP 4 & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\
            (s.e.) & $-$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ \\\\
            DGP 5 & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\
            (s.e.) & $-$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ \\\\
            DGP 6 & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\
            (s.e.) & $-$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ & $(%s)$ \\\\
            \\hline
        \\end{tabular}
    \\end{center}
    """

    tex = tex_dict.copy()

    data = [tex['method'], tex['N'], tex['T']]

    for i in range(6):
        data += ['{0:.2f}'.format(tex['r'][i])]
        data += ['{0:.2f}'.format(tex['PC']['params'][i])]
        data += ['{0:.2f}'.format(tex['PLS']['params'][i])]
        data += ['{0:.2f}'.format(tex['Ridge']['alpha']['params'][i])]
        data += ['{0:.2f}'.format(tex['Ridge']['DOF']['params'][i])]
        data += ['{0:.2f}'.format(tex['LF']['alpha']['params'][i])]
        data += ['{0:.2f}'.format(tex['LF']['DOF']['params'][i])]
        data += ['{0:.2f}'.format(tex['PC']['se'][i])]
        data += ['{0:.2f}'.format(tex['PLS']['se'][i])]
        data += ['{0:.2f}'.format(tex['Ridge']['alpha']['se'][i])]
        data += ['{0:.2f}'.format(tex['Ridge']['DOF']['se'][i])]
        data += ['{0:.2f}'.format(tex['LF']['alpha']['se'][i])]
        data += ['{0:.2f}'.format(tex['LF']['DOF']['se'][i])]

    data = tuple(data)

    string = string%data

    with open(file, 'w') as f:
        f.write(string)