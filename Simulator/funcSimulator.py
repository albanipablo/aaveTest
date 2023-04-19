import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

def runStrategy(yields, pools, chains, protocols, stables, investment, days_to_change, swap_fee, bridge_fee, yield_upside_factor):
    pools = pools[pools.chain.isin(chains)]

    # Filter Protocols
    pools = pools[pools.project.isin(protocols)]

    def filter_stables(row):
        tokens = row['symbol'].split('-')
        return all(s in stables for s in tokens)

    # Filter only Selected Stablecoins
    pools = pools[pools.apply(filter_stables, axis=1)]

    df = pd.merge(pools[['chain','project','symbol','pool']],
                  yields[['pool','timestamp','tvlUsd','apy','apyBase','apyReward']])

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date

    # Yield to % y diaria
    df['apy'] = df.apy / 100 / 365

    #Pivot table
    df['chain-project-symbol'] = df['chain'] + '_' + df['project'] + '_' + df['symbol']
    df = pd.pivot_table(df, values=['tvlUsd', 'apy'], index='timestamp', columns='chain-project-symbol')

    # Multi index to simple index
    df.columns = [f'{col[0]}-{col[1]}' for col in df.columns]

    # Limit dates
    df = df[df.index > pd.to_datetime('2022-04-01').date()]
    df = df[df.index < pd.to_datetime('2023-04-01').date()]

    # Re-calulate yields taking account of our position
    poolNames = df.loc[:,df.columns.str.contains('apy')].columns.tolist()
    poolNames = [s[4:] for s in poolNames]

    for p in poolNames:
        df[f'apy-{p}'] = (df[f'tvlUsd-{p}'] * df[f'apy-{p}']) / (df[f'tvlUsd-{p}'] + investment)

    # Get Max APY
    cols = [col for col in df.columns if col.startswith('apy')]

    df['max_apy'] = df[cols].max(axis=1)
    df['max_apy_pool'] = [s[4:] for s in df[cols].idxmax(axis=1)]

    df[['chain', 'protocol', 'token']] = df['max_apy_pool'].str.split('_', expand=True)

    df['ChangeInChain'] = df['chain'] != df['chain'].shift(1)
    df['ChangeInToken'] = df['token'] != df['token'].shift(1)

    df[['PreviousPool', 'PreviousTimestamp']] = None

    enough_time_passed = []
    same_chain_and_token = []
    upside_gr_bridgeFee = []
    upside_gr_SwapFee = []
    upside_gr_Bridge_and_SwapFee = []

    def updatePrevious():
        df.loc[df.index[i], 'PreviousPool'] = df.at[df.index[i], 'max_apy_pool']

        if df.at[df.index[i], 'PreviousPool'] == df.at[df.index[i-1], 'PreviousPool']:
            df.at[df.index[i], 'PreviousTimestamp'] = df.at[df.index[i-1], 'PreviousTimestamp']
        else:
            df.at[df.index[i], 'PreviousTimestamp'] = df.index[i]


    # Loop through each row of the DataFrame and perform the correction
    for i in range(len(df)):
        if i == 0:
            # If it's the first row, save the current values in all columns
            updatePrevious()
        elif df.at[df.index[i], 'max_apy_pool'] == df.at[df.index[i-1], 'PreviousPool']:
            # If the current pool is still the best
            updatePrevious()
        else:
            # Calculate the elapsed time since the last pool selection and the yield upside
            elapsed_time = df.index[i] - df.at[df.index[i-1], 'PreviousTimestamp']
            yield_upside = df.max_apy[i] - df.iloc[i].loc[f'apy-{df.max_apy_pool[i-1]}']     

            if (~df.at[df.index[i], 'ChangeInToken']) & (~df.at[df.index[i], 'ChangeInChain']):
                # If its the same token and chain
                updatePrevious()
                same_chain_and_token.append(i)

            elif ~df.at[df.index[i], 'ChangeInToken'] & (yield_upside * yield_upside_factor > bridge_fee):  
                # If its the same Token and: yield_upside > cost
                updatePrevious()
                upside_gr_bridgeFee.append(i)

            elif ~df.at[df.index[i], 'ChangeInChain'] & (yield_upside * yield_upside_factor > swap_fee):
                # If its the same Chain and: yield_upside > cost
                updatePrevious()
                upside_gr_SwapFee.append(i)

            elif (yield_upside * yield_upside_factor) > (swap_fee + bridge_fee):
                # If: yield_upside > cost of swaping and bridging
                updatePrevious()
                upside_gr_Bridge_and_SwapFee.append(i)

            elif elapsed_time >= timedelta(days=days_to_change):
                # If the time required have passed, let it change 
                updatePrevious()
                if df.at[df.index[i], 'PreviousPool'] != df.at[df.index[i-1], 'PreviousPool']:
                    enough_time_passed.append(i)

            else:
                # If not, keep the previous pool
                df.loc[df.index[i], 'max_apy_pool'] = df.at[df.index[i-1], 'max_apy_pool']
                df.loc[df.index[i], 'max_apy'] = df.at[df.index[i-1], 'max_apy']

                df.loc[df.index[i], 'PreviousPool'] = df.at[df.index[i-1], 'max_apy_pool']
                df.loc[df.index[i], 'PreviousTimestamp'] = df.at[df.index[i-1], 'PreviousTimestamp']

                df.loc[df.index[i],['chain', 'protocol', 'token']] = df.at[df.index[i], 'max_apy_pool'].split('_')
                df['ChangeInChain'] = df['chain'] != df['chain'].shift(1)
                df['ChangeInToken'] = df['token'] != df['token'].shift(1)

    print(f'Change Reason -> Same Chain and Token = {same_chain_and_token}')
    print(f'Change Reason -> Same Token & Upside > bridgeFee = {upside_gr_bridgeFee}')
    print(f'Change Reason -> Same Chain & Upside > swapFee = {upside_gr_SwapFee}')
    print(f'Change Reason -> Upside > bridgeFee & swapFee = {upside_gr_Bridge_and_SwapFee}')
    print(f'Change Reason -> Enough time passed = {enough_time_passed}')

    # Drop the "PreviousPool" and "PreviousTimestamp" columns
    df.drop(['PreviousPool', 'PreviousTimestamp'], axis=1, inplace=True)

    # Add fees to max_apy
    update_yield = lambda row: row['max_apy'] - \
                                (bridge_fee if row['ChangeInChain'] else 0) - \
                                (swap_fee if row['ChangeInToken'] else 0)

    df['real_apy'] = df.apply(update_yield, axis=1)

    # Cummulative result for NOYA
    df['NoyaStrategy_cum'] = df['real_apy'].cumsum()

    # Cummulative result for Chain / Protocol
    for c in df.chain.unique():
        for p in df.protocol.unique():
            df[f'{c}_{p}_cum'] = df.loc[:, df.columns.str.contains(f"apy-{c}_{p}*")].mean(axis=1).cumsum()

    ax = df.loc[:, df.columns.str.contains('_cum')].rename(columns=lambda x: x[:-4]).plot(figsize=(10, 6))

    ax.get_lines()[0].set_linewidth(3)
    ax.get_lines()[0].set_color('green')

    ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1), fontsize=8, ncol=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4)) # creates 1 row and 3 columns of subplots

    # plot the first pie chart
    axes[0].pie(df.protocol.value_counts().values, labels=df.protocol.value_counts().index, 
                autopct='%1.1f%%', textprops={'fontsize': 8})
    axes[0].set_title('What protocols did the strategy use?', fontsize=10)

    # plot the second pie chart
    axes[1].pie(df.chain.value_counts().values, labels=df.chain.value_counts().index, 
                autopct='%1.1f%%', textprops={'fontsize': 8})
    axes[1].set_title('What Blochchains did the strategy use?', fontsize=10)

    # plot the third pie chart
    axes[2].pie(df.token.value_counts().values, labels=df.token.value_counts().index, 
                autopct='%1.1f%%', textprops={'fontsize': 8})
    axes[2].set_title('What Token did the strategy use?', fontsize=10)

    return df