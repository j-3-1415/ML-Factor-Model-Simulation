# 3. Function to output latex regression table
def out_latex(file):

    ############################################################################
    # Input Notes
    ############################################################################

    num_cols = 2

    # Define the beginning of latex tabular to be put within table definition
    string = "\\renewcommand{\\arraystretch}{0.5}"

    string += "\\begin{center}\n\\begin{table}{" + "c" * num_cols + \
        "}\n" + "\\\\[-1.8ex]\\hline\n"
    string += "Hello & Yes"
    string += "\\\\\n \\hline \\hline \n"

    string += "\\end{table}\n\\end{center}"

    # return(string)

    with open(file, 'w') as f:
        f.write(string)

    return(None)

    full_ind = []
    for key in models.keys():
        indices = [
            "Interacted" + i.split("BY")[0] if 'BY' in i else i for i in models[key].params.index]
        if use_lags:
            indices = [
                "Interacted" + i.split("_")[-1].split("BY")[0] if 'BY' in i else i.split("_")[-1] for i in models[key].params.index]
        full_ind.extend(indices)

    full_ind = set(full_ind)
    full_ind = [col for col in order if col in full_ind]

    num_params = len(full_ind)

    vals = {key: {'params': {}, 'errors': {}, 'pvals': {}}
            for key in models.keys()}

    for key in models.keys():
        indices = [i for i in models[key].params.index]
        params = models[key].params.values
        errors = models[key].std_errors.values
        pvals = models[key].pvalues.values

        num_params = len(set(indices))

        if use_lags:
            params = params[:num_params]
            errors = errors[:num_params]
            pvals = pvals[:num_params]

        vals[key]['params'] = dict(zip(indices, params))
        vals[key]['errors'] = dict(zip(indices, errors))
        vals[key]['pvals'] = dict(zip(indices, pvals))

    if not use_lags:
        lab_list = {i: labs[i] for i in full_ind}
    else:
        lab_list = {i: lag_labs[i] for i in full_ind}

    # Iterate through the number of coefficients
    for i, ind in enumerate(full_ind):
        string += "\\\\"

        string += lab_list[ind]

        for key in models.keys():
            if ind in vals[key]['params'].keys():
                param = str("{:.4f}".format(vals[key]['params'][ind]))
                if vals[key]['pvals'][ind] <= 0.01:
                    star = "$^{***}$"
                elif vals[key]['pvals'][ind] <= 0.05:
                    star = "$^{**}$"
                elif vals[key]['pvals'][ind] <= 0.1:
                    star = "$^{*}$"
                else:
                    star = ""
            else:
                param = ""
                star = ""

            string += " & " + param + star

        string += "\\\\ \n "

        for key in models.keys():

            if ind in vals[key]['params'].keys():
                error = "(" + \
                    str("{:.4f}".format(vals[key]['errors'][ind])) + ")"
            else:
                error = ""
            string += "&" + error

        string += "\n"

    for model in models:
        if not use_lags:
            effects = models[model].included_effects
        else:
            effects = ['Time']
        string += " & " + ", ".join([effect for effect in effects])

    # Include R-Squared in the outputs
    string += "\\\\ R-Squared: "
    string += " & " + \
        "&".join(["%.3f" % model.rsquared
                  for model in models.values()])

    # Include number of observations
    string += "\\\\ Observations: "
    string += " & " + "&".join([str(model.nobs) for model in models.values()])
    if use_lags:
        string += "\\\\ Over-Identification p-Val: "
        string += " & " + \
            "&".join(["%.3f" % model.j_stat.pval for model in models.values()])
        string += "\\\\ AB AR Order " + \
            str(2) + " p-Val: "
        string += " & " + "&".\
            join(["%.3f" % estat(model, 2)
                  for model in models.values()])
        string += "\\\\ Iterations: "
        string += " & " + "&".join([str(model.iterations)
                                    for model in models.values()])

    string += "\\\\ \\hline \\\\ \n \\multicolumn{" + str(
        len(models)) + "}{c}{Robust Standard Errors are Shown in Parentheses} \\\\"

    if use_lags:
        string += "\n \\multicolumn{" + str(len(models)) + "}{c}{Max Lag Depth = " + str(
            model_params['max_lags']) + "} \\\\"

    string += "\\end{longtable}\n\\end{center}"

    # After creating latex string, write to tex file
    with open(file, 'w') as f:
        f.write(string)
    # After creating latex string, write to tex file
    with open(file, 'w') as f:
        f.write(string)