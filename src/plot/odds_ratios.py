import pandas as pd
import numpy as np
import seaborn as sns
import arviz as az
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

def plot_odds_ratio(idata, ax, recode_mapping_reverse,  coef_name, var_name, baseline, survey = None, add_n = False, title = None, num_samples = 2000, max_len_label = 10, plot_type = 'boxplot',
                    violin_width = None, odds_ratio = True, palette = None, alpha = 1, orderd_labels_dict = {}, title_size = None ):
    if palette is None:
        palette = {
            3: sns.color_palette("Blues", n_colors=8)[6],  # High Positive (Dark Blue)
            2: sns.color_palette("Blues", n_colors=8)[4],  # Medium Positive (Medium Blue)
            1: sns.color_palette("Blues", n_colors=8)[2],  # Low Positive (Light Blue)
            0 : sns.color_palette("Grays", n_colors=7)[2],
            -1: sns.color_palette("Reds", n_colors=8)[2],  # Low Negative (Light Red)
            -2: sns.color_palette("Reds", n_colors=8)[4],  # Medium Negative
            -3: sns.color_palette("Reds", n_colors=8)[6],  # High Negative (Dark Red)
        }

    if add_n and survey is None:
        raise ValueError('must provide survey when add_n = True')
    def get_color(df,c):
        perc = 0.1 * 50
        if ( np.percentile(df[c], 10) > 0 ) and (np.percentile(df[c], 5) <= 0):
            return(palette[1])
        elif ( np.percentile(df[c], 5) > 0 ) and (np.percentile(df[c], 0.5) <= 0):
            return(palette[2])
        elif (np.percentile(df[c], 0.5) > 0):
            return(palette[3])
            
        elif (np.percentile(df[c], 90) < 0) and (np.percentile(df[c], 95) >= 0):
            return(palette[-1])
        elif (np.percentile(df[c], 95) < 0) and (np.percentile(df[c], 99.5) >= 0):
            return(palette[-2])
        elif (np.percentile(df[c], 99.5) < 0):
            return(palette[-3])
        else:
            return(palette[0])
    _post = idata.posterior[coef_name].values.reshape(num_samples,-1)
    
    if _post.shape[1] == 1:
        _post = np.hstack([_post, -_post])
    _df = pd.DataFrame(_post, columns = recode_mapping_reverse[var_name].values() )

    if var_name in orderd_labels_dict:
        orderd_labels = orderd_labels_dict[var_name]
        _df = _df[orderd_labels]
        

    original_columns = _df.columns
    truncated_columns = [col[:max_len_label] for k,col in enumerate(_df.columns)]
    if len(set(original_columns)) != len(set(truncated_columns)):
        truncated_columns = [col[:max_len_label] + f' c{k}' for k,col in enumerate(_df.columns)]
    _df.columns = truncated_columns

    _df = _df.drop(columns = [baseline]) - np.array(_df[baseline])[:, None]

    
    my_pal = {c : get_color(_df,c) for c in _df.columns}
    
    _df = np.exp(_df)

    if plot_type == 'boxplot':
        sns.boxplot(data = _df, orient="h", ax = ax, palette=my_pal)
    elif plot_type == 'violin':
        if violin_width is None:
            sns.violinplot(data=_df, orient="h", ax=ax, palette=my_pal, linewidth=0.5, linecolor='black', inner= 'box', alpha = alpha)# inner one of: 'box', 'stick', 'point', None
        else:
            sns.violinplot(data=_df, orient="h", ax=ax, palette=my_pal, linewidth=0.5, linecolor='black', inner= 'box', density_norm='area', width=violin_width, alpha = alpha)
        
    else:
        raise ValueError('non valid plot_type')

    if add_n:
        num_n = dict( survey[var_name].value_counts() )
        num_n = {k : f'n={v}' for k,v in num_n.items()}
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels([num_n[col] for col in _df.columns])
        ax2.grid(False) 

        
        ax2.tick_params(length=0)

    if odds_ratio:
        ax.axvline(x = 1, c = 'dimgray', linewidth = 2, zorder = 0)
    else:    
        ax.axvline(x = 0, c = 'dimgray', linewidth = 2, zorder = 0)
    if title is not None:
        if add_n:
            title = f'{title} ({num_n[baseline]})'
        if title_size:
            ax.set_title(title, size = title_size)
        else:
            ax.set_title(title)
    

def plot_odds_ratio_rct(idata, ax, recode_mapping_reverse,  coef_name, var_name, baseline, survey = None, add_n = False, title = None, num_samples = 2000, N_treat = 7, control = 0, treatment = 6,
                        max_len_label = 10, plot_type = 'violin', violin_width = None, odds_ratio = True,
                        palette = None, alpha = 1, orderd_labels_dict = {}, title_size = None, _df = None ):

    if add_n and survey is None:
        raise ValueError('must provide survey when add_n = True')
    def get_color(df,c):
        perc = 0.1 * 50
        if ( np.percentile(df[c], 10) > 0 ) and (np.percentile(df[c], 5) <= 0):
            return(palette[1])
        elif ( np.percentile(df[c], 5) > 0 ) and (np.percentile(df[c], 0.5) <= 0):
            return(palette[2])
        elif (np.percentile(df[c], 0.5) > 0):
            return(palette[3])
            
        elif (np.percentile(df[c], 90) < 0) and (np.percentile(df[c], 95) >= 0):
            return(palette[-1])
        elif (np.percentile(df[c], 95) < 0) and (np.percentile(df[c], 99.5) >= 0):
            return(palette[-2])
        elif (np.percentile(df[c], 99.5) < 0):
            return(palette[-3])
        else:
            return(palette[0])

    if palette is None:
        palette = {
            3: sns.color_palette("Blues", n_colors=8)[6],  # High Positive (Dark Blue)
            2: sns.color_palette("Blues", n_colors=8)[4],  # Medium Positive (Medium Blue)
            1: sns.color_palette("Blues", n_colors=8)[2],  # Low Positive (Light Blue)
            0 : sns.color_palette("Grays", n_colors=7)[2],
            -1: sns.color_palette("Reds", n_colors=8)[2],  # Low Negative (Light Red)
            -2: sns.color_palette("Reds", n_colors=8)[4],  # Medium Negative
            -3: sns.color_palette("Reds", n_colors=8)[6],  # High Negative (Dark Red)
        }

    posterior_post = idata['posterior'][coef_name].values.reshape(num_samples, N_treat, -1)
    posterior_pre = idata['posterior'][f'{coef_name}_pre'].values.reshape(num_samples, -1)
    
    if posterior_pre.shape[1] == 1:
        posterior_pre = np.hstack([posterior_pre, -posterior_pre])

    if posterior_post.shape[-1] == 1:
        _posterior_post = np.zeros((posterior_post.shape[0], posterior_post.shape[1], 2))
        
        _posterior_post[:, :, 0] = posterior_post[:, :, 0]
        _posterior_post[:, :, 1] = -posterior_post[:, :, 0]
        posterior_post = _posterior_post.copy()
    
    posterior_control = posterior_post[:,control,:]
    posterior_treatment = posterior_post[:,treatment,:]

    log_odds_ratio_control = pd.DataFrame(posterior_control, columns = recode_mapping_reverse[var_name].values() )
    
    log_odds_ratio_treatment = pd.DataFrame(posterior_treatment, columns = recode_mapping_reverse[var_name].values() )

    if var_name in orderd_labels_dict:
        orderd_labels = orderd_labels_dict[var_name]
        log_odds_ratio_treatment = log_odds_ratio_treatment[orderd_labels]
        log_odds_ratio_control = log_odds_ratio_control[orderd_labels]
    
    log_odds_ratio_treatment = log_odds_ratio_treatment.drop(columns = [baseline]) - np.array(log_odds_ratio_treatment[baseline])[:, None]

    log_odds_ratio_control = log_odds_ratio_control.drop(columns = [baseline]) - np.array(log_odds_ratio_control[baseline])[:, None]
    
    log_odds_diff = log_odds_ratio_treatment - log_odds_ratio_control
    
    my_pal = {c : get_color(log_odds_diff,c) for c in log_odds_diff.columns}
    
    ratio_of_odds_ratio = np.exp(log_odds_diff)

    if odds_ratio:
    
        if plot_type == 'boxplot':
            sns.boxplot(data = ratio_of_odds_ratio, orient="h", ax = ax, palette=my_pal)
        elif plot_type == 'violin':
            if violin_width is None:
                sns.violinplot(data = ratio_of_odds_ratio, orient="h", ax=ax, palette=my_pal, linewidth=0.5, linecolor='black', inner= 'box')# inner one of: 'box', 'stick', 'point', None
            else:
                sns.violinplot(data = ratio_of_odds_ratio, orient="h", ax=ax, palette=my_pal, linewidth=0.5, linecolor='black', inner= 'box', density_norm='area', width=violin_width)
            
        else:
            raise ValueError('non valid plot_type')

    else:
        if plot_type == 'boxplot':
            sns.boxplot(data = log_odds_diff, orient="h", ax = ax, palette=my_pal)
        elif plot_type == 'violin':
            if violin_width is None:
                sns.violinplot(data = log_odds_diff, orient="h", ax=ax, palette=my_pal, linewidth=0.5, linecolor='black', inner= 'box')# inner one of: 'box', 'stick', 'point', None
            else:
                sns.violinplot(data = log_odds_diff, orient="h", ax=ax, palette=my_pal, linewidth=0.5, linecolor='black', inner= 'box', density_norm='area', width=violin_width)
            
        else:
            raise ValueError('non valid plot_type')
    
    if add_n:
        num_n = dict( survey[var_name].value_counts() )
        num_n = {k : f'n={v}' for k,v in num_n.items()}
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels([num_n[col] for col in _df.columns])
        ax2.grid(False) 

        
        ax2.tick_params(length=0)

    if odds_ratio:
        ax.axvline(x = 1, c = 'dimgray', linewidth = 2, zorder = 0)
    else:    
        ax.axvline(x = 0, c = 'dimgray', linewidth = 2, zorder = 0)
    if title is not None:
        if add_n:
            title = f'{title} ({num_n[baseline]})'
        if title_size:
            ax.set_title(title, size = title_size)
        else:
            ax.set_title(title)


def plot_odds_ratio_bes_soc(idata_bes, idata_soc, soc_dems, ax, recode_mapping_reverse,  coef_name, var_name, baseline, survey = None, add_n = False, title = None, num_samples = 2000, max_len_label = 10, plot_type = 'boxplot',
                    violin_width = None, odds_ratio = True, palette = None, alpha = 1, orderd_labels_dict = {}, title_size = None, hatch_width = 1.0, hatch_symbol = '/', hatch_color = 'white' ):
    
    from matplotlib.collections import PolyCollection
    import matplotlib as mpl
    mpl.rcParams['hatch.linewidth'] = hatch_width

    if palette is None:
        palette = {
            3: sns.color_palette("Blues", n_colors=8)[6],  # High Positive (Dark Blue)
            2: sns.color_palette("Blues", n_colors=8)[4],  # Medium Positive (Medium Blue)
            1: sns.color_palette("Blues", n_colors=8)[2],  # Low Positive (Light Blue)
            0 : sns.color_palette("Grays", n_colors=7)[2],
            -1: sns.color_palette("Reds", n_colors=8)[2],  # Low Negative (Light Red)
            -2: sns.color_palette("Reds", n_colors=8)[4],  # Medium Negative
            -3: sns.color_palette("Reds", n_colors=8)[6],  # High Negative (Dark Red)
        }
    if add_n and survey is None:
        raise ValueError('must provide survey when add_n = True')
    def get_color(df,c):
        perc = 0.1 * 50
        if ( np.percentile(df[c], 10) > 0 ) and (np.percentile(df[c], 5) <= 0):
            return(palette[1])
        elif ( np.percentile(df[c], 5) > 0 ) and (np.percentile(df[c], 0.5) <= 0):
            return(palette[2])
        elif (np.percentile(df[c], 0.5) > 0):
            # return('deepskyblue')
            return(palette[3])
            
        elif (np.percentile(df[c], 90) < 0) and (np.percentile(df[c], 95) >= 0):
            return(palette[-1])
        elif (np.percentile(df[c], 95) < 0) and (np.percentile(df[c], 99.5) >= 0):
            return(palette[-2])
        elif (np.percentile(df[c], 99.5) < 0):
            return(palette[-3])
        else:
            return(palette[0])
    _post = idata_bes.posterior[coef_name].values.reshape(num_samples,-1)
    
    if _post.shape[1] == 1:
        _post = np.hstack([_post, -_post])
    _df = pd.DataFrame(_post, columns = recode_mapping_reverse[var_name].values() )

    if var_name in orderd_labels_dict:
        orderd_labels = orderd_labels_dict[var_name]
        _df = _df[orderd_labels]
        
    if var_name in soc_dems:
        _post_soc = idata_soc.posterior[coef_name].values.reshape(num_samples,-1)

        if _post_soc.shape[1] == 1:
            _post_soc = np.hstack([_post_soc, -_post_soc])
        _df_soc = pd.DataFrame(_post_soc, columns = recode_mapping_reverse[var_name].values() )
    
        if var_name in orderd_labels_dict:
            orderd_labels = orderd_labels_dict[var_name]
            _df_soc = _df_soc[orderd_labels]

    original_columns = _df.columns
    truncated_columns = [col[:max_len_label] for k,col in enumerate(_df.columns)]
    if len(set(original_columns)) != len(set(truncated_columns)):
        truncated_columns = [col[:max_len_label] + f' c{k}' for k,col in enumerate(_df.columns)]
    _df.columns = truncated_columns
    _df = _df.drop(columns = [baseline]) - np.array(_df[baseline])[:, None]
    
    my_pal = {c : get_color(_df,c) for c in _df.columns}
    
    _df = np.exp(_df)

    if var_name in soc_dems:
        _df_soc.columns = truncated_columns
        _df_soc = _df_soc.drop(columns = [baseline]) - np.array(_df_soc[baseline])[:, None]
        my_pal_soc = {c : get_color(_df_soc,c) for c in _df_soc.columns}
        _df_soc = np.exp(_df_soc)

    if var_name not in soc_dems:
        if plot_type == 'boxplot':
            sns.boxplot(data = _df, orient="h", ax = ax, palette=my_pal)
        elif plot_type == 'violin':
            if violin_width is None:
                sns.violinplot(data=_df, orient="h", ax=ax, palette=my_pal, linewidth=0.5, linecolor='black', inner= 'box', alpha = alpha)# inner one of: 'box', 'stick', 'point', None
            else:
                sns.violinplot(data=_df, orient="h", ax=ax, palette=my_pal, linewidth=0.5, linecolor='black', inner= 'box', density_norm='area', width=violin_width, alpha = alpha)
            
        else:
            raise ValueError('non valid plot_type')
    else:
        
        # Concatenate the data
        _df['model'] = 'with_besd'
        _df_soc['model'] = 'without_besd'
        combined_data = pd.concat([_df, _df_soc], axis=0, ignore_index=True)
        
        # Melt the combined DataFrame to long format
        long_data = combined_data.melt(id_vars='model', var_name='parameter', value_name='value')
        
        sns.violinplot(data=long_data, x='value', y='parameter', hue='model', native_scale=True, common_norm=True,
                       ax=ax, linewidth=0.5, linecolor='black', inner= 'box', density_norm = 'width', # density_norm='area', 
                       width=violin_width, alpha = alpha, legend=False) #, palette={"with_besd": "red", "without_besd": "blue"}
        for ind, violin in enumerate(ax.findobj(PolyCollection)):
            if ind % 2 == 0:
                _color = list( my_pal.values() )[ind//2]
            else:
                _color = list( my_pal_soc.values() )[ind//2]
            violin.set_facecolor(_color)
            
            # Apply hatching to 'post2' violins (odd indices)
            if ind % 2 != 0:
                #violin.set_hatch('/')  # Apply hatching
                # violin.set_edgecolor('black') 
                violin._hatch_color = mpl.colors.to_rgba("white")
                violin.set_hatch(hatch_symbol) # options are {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
                violin._hatch_color = mpl.colors.to_rgba("white")
                violin.set_edgecolor(hatch_color) 

        #########################
        

    if add_n:
        num_n = dict( survey[var_name].value_counts() )
        num_n = {k : f'n={v}' for k,v in num_n.items()}
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels([num_n[col] for col in _df.columns if col != 'model'])
        ax2.grid(False) 

        
        ax2.tick_params(length=0)

    if odds_ratio:
        ax.axvline(x = 1, c = 'dimgray', linewidth = 2, zorder = 0)
    else:    
        ax.axvline(x = 0, c = 'dimgray', linewidth = 2, zorder = 0)
    if title is not None:
        if add_n:
            title = f'{title} ({num_n[baseline]})'
        if title_size:
            ax.set_title(title, size = title_size)
        else:
            ax.set_title(title)
    


def add_custom_legend(fig, anchor_x = 0.525, anchor_y = 0.03, framealpha = 0.5, fontsize=14, title_fontsize=16, palette = None, alpha = 1.0, labels = None ):
    if palette is None:
        palette = {
            3: sns.color_palette("Blues", n_colors=8)[6],  # High Positive (Dark Blue)
            2: sns.color_palette("Blues", n_colors=8)[4],  # Medium Positive (Medium Blue)
            1: sns.color_palette("Blues", n_colors=8)[2],  # Low Positive (Light Blue)
            0 : sns.color_palette("Grays", n_colors=7)[2],
            -1: sns.color_palette("Reds", n_colors=8)[2],  # Low Negative (Light Red)
            -2: sns.color_palette("Reds", n_colors=8)[4],  # Medium Negative
            -3: sns.color_palette("Reds", n_colors=8)[6],  # High Negative (Dark Red)
        }

    palette = {k: mcolors.to_rgba(color, alpha) for k, color in palette.items()}

    if labels is None:
        legend_labels = {
            palette[3]: '99.5% of posterior odds ratio above 1',
            palette[2]: '95% of posterior odds ratio above 1',
            palette[1]: '90% of posterior odds ratio above 1',
            palette[0]: 'At least 10% of posterior odds ratio below and 10% above 1',
            palette[-1]: '90% of posterior odds ratio below 1',
            palette[-2]: '95% of posterior odds ratio below 1',
            palette[-3]: '99.5% of posterior odds ratio below 1',
        }

    else:
        legend_labels = {
            palette[3]: labels[3],
            palette[2]: labels[2],
            palette[1]: labels[1],
            palette[0]: labels[0],
            palette[-1]: labels[-1],
            palette[-2]: labels[-2],
            palette[-3]: labels[-3],
        }
        

    patches = [mpatches.Patch(color=color, label=label) for color, label in legend_labels.items()]
    
    # Adjust legend position with bbox_to_anchor
    fig.legend(handles=patches, loc='lower center', title='Color Legend',  bbox_to_anchor=(anchor_x, anchor_y),
               fontsize=fontsize,               # Adjust font size for legend labels
               title_fontsize=title_fontsize,          # Adjust font size for the legend title
               frameon=True,               # Add a frame around the legend (optional)
               borderaxespad=1.2,
               framealpha = framealpha)