import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit





def runStrategyBacktest(investment, strategy, yield_upside_factor=7, maxPoolsSplit=3, printAll=False):

    # Read Files
    yields = pd.read_csv('yields.csv.gz')    # defiLlama/yields.csv
    pools = pd.read_csv('yieldPools.csv.gz') # defiLlama/yieldPools.csv

    # Discard USDC depeg days
    yields['timestamp'] = pd.to_datetime(yields['timestamp']).dt.date
    yields = yields[(yields.timestamp != pd.to_datetime('2023-03-12').date()) & (yields.timestamp != pd.to_datetime('2023-03-11').date())]

    yields = yields[yields.apy < 100]
    ##################################################

    ###### Set Up Chains / Tokens / Protocols for every Strategy

    chains = ['Ethereum', 'Arbitrum', 'Polygon', 'Avalanche','BSC', 'Optimism', 'Gnosis','Fantom']

    ### Stable
    #Low
    tokensSL = ['USDC', 'USDT', 'DAI', 'BUSD', 'LUSD', 'TUSD', 'DOLA', 'FRAX', '3CRV'] 
    protocolsSL = ['aave-v2', 'aave-v3', 'aura', 'balancer-v2', 'compound', 'curve', 'uniswap-v2', 'uniswap-v3',  'yearn-finance']

    # Medium
    tokensSM = ['USDC', 'USDT', 'DAI', 'BUSD', 'LUSD', 'TUSD', 'DOLA', 'FRAX', '3CRV', 'USDN', 'USDJ', 'GUSD', 'MAI', 'HUSD'] 

    protocolsSM = ['aave-v2', 'aave-v3', 'aura', 'balancer-v2', 'compound', 'curve', 'uniswap-v2', 'uniswap-v3',  'yearn-finance',
                 'alpaca-leveraged-yield-farming', 'alpaca-lending', 'angle', 'beefy', 'convex-finance', 'frax', 'goldfinch',
                 'hop-protocol', 'idle', 'justlend', 'morpho-aave', 'morpho-compound', 'pancakeswap-amm', 'pancakeswap-amm-v3',
                 'ribbon','stakewise', 'stargate', 'velodrome', 'venus']

    #High
    tokensSH = ['USDC', 'USDT', 'DAI', 'BUSD', 'LUSD', 'TUSD', 'DOLA', 'FRAX', '3CRV', 
                'USDN', 'USDJ', 'GUSD', 'MAI', 'HUSD', 'MIM', 'USP' 'ALUSD', 'OUSD', 'SUSD']

    protocolsSH = ['aave-v2', 'aave-v3', 'aura', 'balancer-v2', 'compound', 'curve', 'uniswap-v2', 'uniswap-v3',  'yearn-finance',
                 'alpaca-leveraged-yield-farming', 'alpaca-lending', 'angle', 'beefy', 'convex-finance', 'frax', 'goldfinch',
                 'hop-protocol', 'idle', 'justlend', 'morpho-aave', 'morpho-compound', 'pancakeswap-amm', 'pancakeswap-amm-v3',
                 'ribbon','stakewise', 'stargate', 'velodrome', 'venus', 
                 'acryptos', 'allbridge', 'arbor-finance', 'archimedes-finance', 'arrakis-v1', 'benddao', 'camelot-v2', 
                 'concentrator', 'conic-finance', 'deri-protocol', 'flashstake', 'gains-network', 'gamma', 'gmd-protocol',
                 'iron-bank', 'moonwell-artemis', 'pendle', 'premia', 'reaper-farm', 'sherlock', 'sommelier', 'stakedao',
                 'sturdy', 'synapse', 'tangible', 'tectonic', 'thena', 'trader-joe-dex', 'trader-joe-lend', 'wombex-finance',
                 'woofi-earn']

    ### ETH

    # Low
    tokensEL = ['WETH', 'ETH', 'STETH', 'CBETH', 'RETH', 'SFRXETH', 'ANKRETH', 'SETH2', 'SGETH', 'NETH'] # YETH (not yet in DefiLlama)

    protocolsEL = ['aave-v2', 'aave-v3', 'aura', 'balancer-v2', 'compound', 'curve', 'uniswap-v2', 'uniswap-v3',  'yearn-finance',
                'lido', 'rocket-pool']

    # Medium
    tokensEM = ['WETH', 'ETH', 'STETH', 'CBETH', 'RETH', 'SFRXETH', 'ANKRETH', 'SETH2', 'SGETH', 'NETH'] # YETH (not yet in DefiLlama)

    protocolsEM = ['aave-v2', 'aave-v3', 'aura', 'balancer-v2', 'compound', 'curve', 'uniswap-v2', 'uniswap-v3',  'yearn-finance',
                 'lido', 'rocket-pool', 'coinbase-wrapped-staked-eth', 'stakewise', 'ankr',
                 'alpaca-leveraged-yield-farming', 'alpaca-lending', 'angle', 'beefy', 'convex-finance', 'frax', 'goldfinch',
                 'hop-protocol', 'idle', 'justlend', 'morpho-aave', 'morpho-compound', 'pancakeswap-amm', 'pancakeswap-amm-v3',
                 'ribbon','stakewise', 'stargate', 'velodrome', 'venus']

    # High
    tokensEH = ['WETH', 'ETH', 'STETH', 'CBETH', 'RETH', 'SFRXETH', 'ANKRETH', 'SETH2', 'SGETH', 'NETH'] # YETH (not yet in DefiLlama)

    protocolsEH = ['aave-v2', 'aave-v3', 'aura', 'balancer-v2', 'compound', 'curve', 'uniswap-v2', 'uniswap-v3',  'yearn-finance',
                 'lido', 'rocket-pool', 'coinbase-wrapped-staked-eth', 'stakewise', 'ankr',
                 'alpaca-leveraged-yield-farming', 'alpaca-lending', 'angle', 'beefy', 'convex-finance', 'frax', 'goldfinch',
                 'hop-protocol', 'idle', 'justlend', 'morpho-aave', 'morpho-compound', 'pancakeswap-amm', 'pancakeswap-amm-v3',
                 'ribbon','stakewise', 'stargate', 'velodrome', 'venus', 
                 'acryptos', 'allbridge', 'arbor-finance', 'archimedes-finance', 'arrakis-v1', 'benddao', 'camelot-v2', 
                 'concentrator', 'conic-finance', 'deri-protocol', 'flashstake', 'gains-network', 'gamma', 'gmd-protocol',
                 'iron-bank', 'moonwell-artemis', 'pendle', 'premia', 'reaper-farm', 'sherlock', 'sommelier', 'stakedao',
                 'sturdy', 'synapse', 'tangible', 'tectonic', 'thena', 'trader-joe-dex', 'trader-joe-lend', 'wombex-finance',
                 'woofi-earn']

    ##################################################

    # Define Costs: Bridge Fees, Swap Fees and Gas

    bridgeFees = pd.DataFrame(columns = ['fromChain', 'toChain', 'bridgeFee'])

    fees = [{'fromChain':'Ethereum', 'toChain':'Polygon', 'bridgeFee':'0'},
            {'fromChain':'Ethereum', 'toChain':'BSC', 'bridgeFee':'0.0001'},
            {'fromChain':'Ethereum', 'toChain':'Avalanche', 'bridgeFee':'0.0001'},
            {'fromChain':'Ethereum', 'toChain':'Arbitrum', 'bridgeFee':'0'},
            {'fromChain':'Ethereum', 'toChain':'Optimism', 'bridgeFee':'0'},
            {'fromChain':'Ethereum', 'toChain':'Fantom', 'bridgeFee':'0.0001'},
            {'fromChain':'Polygon', 'toChain':'Ethereum', 'bridgeFee':'0.0002'},
            {'fromChain':'Polygon', 'toChain':'BSC', 'bridgeFee':'0.0002'},
            {'fromChain':'Polygon', 'toChain':'Avalanche', 'bridgeFee':'0.0002'},
            {'fromChain':'Polygon', 'toChain':'Arbitrum', 'bridgeFee':'0.0002'},
            {'fromChain':'Polygon', 'toChain':'Optimism', 'bridgeFee':'0.0002'},
            {'fromChain':'Polygon', 'toChain':'Fantom', 'bridgeFee':'0.0002'},
            {'fromChain':'Avalanche', 'toChain':'Ethereum', 'bridgeFee':'0.0003'},
            {'fromChain':'Avalanche', 'toChain':'Polygon', 'bridgeFee':'0.0003'},
            {'fromChain':'Avalanche', 'toChain':'BSC', 'bridgeFee':'0.0003'},
            {'fromChain':'Avalanche', 'toChain':'Arbitrum', 'bridgeFee':'0.0003'},
            {'fromChain':'Avalanche', 'toChain':'Optimism', 'bridgeFee':'0.0003'},
            {'fromChain':'Avalanche', 'toChain':'Fantom', 'bridgeFee':'0.0003'},
            {'fromChain':'Arbitrum', 'toChain':'Ethereum', 'bridgeFee':'0.0001'},
            {'fromChain':'Arbitrum', 'toChain':'Polygon', 'bridgeFee':'0.0002'},
            {'fromChain':'Arbitrum', 'toChain':'BSC', 'bridgeFee':'0.0002'},
            {'fromChain':'Arbitrum', 'toChain':'Avalanche', 'bridgeFee':'0.0002'},
            {'fromChain':'Arbitrum', 'toChain':'Optimism', 'bridgeFee':'0.0002'},
            {'fromChain':'Arbitrum', 'toChain':'Fantom', 'bridgeFee':'0.0002'},
            {'fromChain':'Optimism', 'toChain':'Ethereum', 'bridgeFee':'0.0001'},
            {'fromChain':'Optimism', 'toChain':'Polygon', 'bridgeFee':'0.0002'},
            {'fromChain':'Optimism', 'toChain':'BSC', 'bridgeFee':'0.0002'},
            {'fromChain':'Optimism', 'toChain':'Avalanche', 'bridgeFee':'0.0002'},
            {'fromChain':'Optimism', 'toChain':'Arbitrum', 'bridgeFee':'0.0002'},
            {'fromChain':'Optimism', 'toChain':'Fantom', 'bridgeFee':'0.0002'},
            {'fromChain':'Fantom', 'toChain':'Ethereum', 'bridgeFee':'0.0003'},
            {'fromChain':'Fantom', 'toChain':'Polygon', 'bridgeFee':'0.0003'},
            {'fromChain':'Fantom', 'toChain':'BSC', 'bridgeFee':'0.0003'},
            {'fromChain':'Fantom', 'toChain':'Avalanche', 'bridgeFee':'0.0003'},
            {'fromChain':'Fantom', 'toChain':'Arbitrum', 'bridgeFee':'0.0003'},
            {'fromChain':'Fantom', 'toChain':'Optimism', 'bridgeFee':'0.0003'}]

    for l in fees:
        bridgeFees = pd.concat([bridgeFees, pd.Series(l).to_frame().T])

    bridgeFees['bridgeFee'] = pd.to_numeric(bridgeFees.bridgeFee)

    ############

    swapFees = 0.04 / 100

    ############

    gasFees = pd.DataFrame(columns = ['chain', 'gasUsd'])

    gas = [{'chain':'Ethereum', 'gasUsd':'50'},
            {'chain':'Polygon', 'gasUsd':'1'},
            {'chain':'Avalanche', 'gasUsd':'2'},
            {'chain':'Arbitrum', 'gasUsd':'2'},
            {'chain':'Optimism', 'gasUsd':'1'},
            {'chain':'Fantom', 'gasUsd':'1'}]

    for g in gas:
        gasFees = pd.concat([gasFees, pd.Series(g).to_frame().T])

    gasFees['gasUsd'] = pd.to_numeric(gasFees.gasUsd)

    ##################################################

    if strategy == 'StableLowRisk':
        tokens, protocols = tokensSL, protocolsSL
    elif strategy == 'StableMediumRisk':
        tokens, protocols = tokensSM, protocolsSM
    elif strategy == 'StableHighRisk':
        tokens, protocols = tokensSH, protocolsSH
    elif strategy == 'EthLowRisk':
        tokens, protocols = tokensEL, protocolsEL
    elif strategy == 'EthMediumRisk':
        tokens, protocols = tokensEM, protocolsEM
    elif strategy == 'EthHighRisk':
        tokens, protocols = tokensEH, protocolsEH

    lgbmodel = lgb.Booster(model_file='lgb-Class-70var.txt')

    pools = pools[pools.chain.isin(chains)]

    # Filter Protocols
    pools = pools[pools.project.isin(protocols)]

    def filter_tokens(row):
        t = row['symbol'].split('-')
        return all(s in tokens for s in t)

    # Filter only Selected Tokens
    pools = pools[pools.apply(filter_tokens, axis=1)]

    df = pd.merge(pools[['chain','project','symbol','pool']],
                  yields[['pool','timestamp','tvlUsd','apy','apyBase','apyReward']])

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date

    # Yield to % y daily
    df['apy'] = df.apy / 100 / 365

    # Get Unique Chains & Projects
    chainList = df.chain.unique()
    projectList = df.project.unique()

    # Pivot table
    df['chain-project-symbol'] = df['chain'] + '_' + df['project'] + '_' + df['symbol']
    df = pd.pivot_table(df, values=['tvlUsd', 'apy'], index='timestamp', columns='chain-project-symbol')

    # Multi index to simple index
    df.columns = [f'{col[0]}-{col[1]}' for col in df.columns]

    # Limit dates
    df = df[df.index > pd.to_datetime('2022-05-15').date()]
    #df = df[df.index < pd.to_datetime('2023-04-01').date()]

    def optimizePosition(aprs, tlvs, investment=investment):
        x0 = [1 / len(aprs) for _ in range(len(aprs))]
        bounds = [(0, 1) for _ in range(len(aprs))]

        def constraint(x):
            return np.sum(x) - 1

        def objective(coefs, tlvs, aprs, value):
            return -sum(s * t / (s + c * value) * c * value for c, s, t in zip(coefs, tlvs, aprs))

        return minimize(objective, x0, args=(tlvs, aprs, investment), bounds=bounds, constraints={'type': 'eq', 'fun': constraint})

    def createDataset(bestPools):
        baseCoin = pd.DataFrame()
        for col in bestPools.pool:
            col = f'apy-{col}'
            chain, protocol, token = col[4:].split('_')

            for d in [1, 3, 7, 14, 30]:
                # Same Pool Variation
                baseCoin.loc[col,f'apy-Var{d}Day'] = df.loc[df.index[i-d], col] / df.loc[df.index[i], col] - 1
                baseCoin.loc[col,f'tvlUsd-Var{d}Day'] = df.loc[df.index[i-d], f"tvlUsd-{col[4:]}"] / df.loc[df.index[i], f"tvlUsd-{col[4:]}"] - 1

                # Average of the chain/protocol/token Variation
                baseCoin.loc[col,f'apy-chain-mean-Var{d}Day'] = df.loc[df.index[i-d], df.columns.str.contains(f"apy.*{chain}")].mean() / \
                                                              df.loc[df.index[i], df.columns.str.contains(f"apy.*{chain}")].mean() - 1
                baseCoin.loc[col,f'tvlUsd-chain-mean-Var{d}Day'] = df.loc[df.index[i-d], df.columns.str.contains(f"tvlUsd.*{chain}")].mean() / \
                                                              df.loc[df.index[i], df.columns.str.contains(f"tvlUsd.*{chain}")].mean() - 1
                baseCoin.loc[col,f'apy-protocol-mean-Var{d}Day'] = df.loc[df.index[i-d], df.columns.str.contains(f"apy.*{protocol}")].mean() / \
                                                              df.loc[df.index[i], df.columns.str.contains(f"apy.*{protocol}")].mean() - 1
                baseCoin.loc[col,f'tvlUsd-protocol-mean-Var{d}Day'] = df.loc[df.index[i-d], df.columns.str.contains(f"tvlUsd.*{protocol}")].mean() / \
                                                              df.loc[df.index[i], df.columns.str.contains(f"tvlUsd.*{protocol}")].mean() - 1
                baseCoin.loc[col,f'apy-token-mean-Var{d}Day'] = df.loc[df.index[i-d], df.columns.str.contains(f"apy.*{token}")].mean() / \
                                                              df.loc[df.index[i], df.columns.str.contains(f"apy.*{token}")].mean() - 1
                baseCoin.loc[col,f'tvlUsd-token-mean-Var{d}Day'] = df.loc[df.index[i-d], df.columns.str.contains(f"tvlUsd.*{token}")].mean() / \
                                                              df.loc[df.index[i], df.columns.str.contains(f"tvlUsd.*{token}")].mean() - 1

                # Max Yield of the chain/protocol/token Variation
                baseCoin.loc[col,f'apy-chain-max-Var{d}Day'] = df.loc[df.index[i-d], df.columns.str.contains(f"apy.*{chain}")].max() / \
                                                              df.loc[df.index[i], df.columns.str.contains(f"apy.*{chain}")].max() - 1
                baseCoin.loc[col,f'apy-protocol-max-Var{d}Day'] = df.loc[df.index[i-d], df.columns.str.contains(f"apy.*{protocol}")].max() / \
                                                              df.loc[df.index[i], df.columns.str.contains(f"apy.*{protocol}")].max() - 1
                baseCoin.loc[col,f'apy-token-max-Var{d}Day'] = df.loc[df.index[i-d], df.columns.str.contains(f"apy.*{token}")].max() / \
                                                              df.loc[df.index[i], df.columns.str.contains(f"apy.*{token}")].max() - 1

                # Sum of TVL of the chain/protocol/token Variation
                baseCoin.loc[col,f'tvlUsd-chain-sum-Var{d}Day'] = df.loc[df.index[i-d], df.columns.str.contains(f"tvlUsd.*{chain}")].sum() / \
                                                              df.loc[df.index[i], df.columns.str.contains(f"tvlUsd.*{chain}")].sum() - 1
                baseCoin.loc[col,f'tvlUsd-protocol-sum-Var{d}Day'] = df.loc[df.index[i-d], df.columns.str.contains(f"tvlUsd.*{protocol}")].sum() / \
                                                              df.loc[df.index[i], df.columns.str.contains(f"tvlUsd.*{protocol}")].sum() - 1
                baseCoin.loc[col,f'tvlUsd-token-sum-Var{d}Day'] = df.loc[df.index[i-d], df.columns.str.contains(f"tvlUsd.*{token}")].sum() / \
                                                              df.loc[df.index[i], df.columns.str.contains(f"tvlUsd.*{token}")].sum() - 1
        return baseCoin.fillna(0)


    # Loop through each row of the DataFrame and perform the correction
    for i in range(len(df)):
        print(f'Doing: {df.index[i]} {" "*20}', end='\r')
        bestPools = pd.DataFrame()
        y = df.loc[df.index[i], df.columns.str.startswith('apy')].sort_values(ascending=False)[:maxPoolsSplit]
        bestPools['pool'] = y.index.str[4:]
        bestPools['apr'] = y.values
        bestPools['tlv'] = df["tvlUsd-" + y.index.str[4:]].loc[df.index[i]].values
        bestPools['position'] = optimizePosition(bestPools['apr'], bestPools['tlv']).x.round(2)
        bestPools['newApr'] = (bestPools.tlv * bestPools.apr) / (bestPools.tlv + investment * bestPools.position)
        bestPools[['chain', 'protocol', 'token']] = bestPools['pool'].str.split('_', expand=True)

        for ix, v in bestPools.iterrows():
            df.loc[df.index[i], f'position_{ix+1}'] = v.position
            df.loc[df.index[i], f'bestPool_{ix+1}'] = v.pool
            df.loc[df.index[i], f'bestApr_{ix+1}'] = v.apr

        df['max_apy'] = (bestPools.newApr * bestPools.position).sum()

        if i == 0:
            # If it's the first row
            previousBestPools = bestPools
            df.at[df.index[i], 'real_apy'] = df.at[df.index[i], 'max_apy']
            investment *= (1 + df.at[df.index[i], 'real_apy'])
        else:
            # Calculate currentApr & yield_upside
            pix = 0
            currentApr = 0
            for p in df.loc[df.index[i-1], df.columns.str.startswith('bestPool_')]:
                pix += 1
                currentApr += df.loc[df.index[i], f'apy-{p}'] * df.loc[df.index[i-1], f'position_{pix}']
            yield_upside = df.max_apy[i] - currentApr

            # Define Bridge Fee
            s1 = previousBestPools .groupby('chain').position.sum()
            s2 = bestPools.groupby('chain').position.sum()
            bfee = s1[s1 > 0].combine(s2[s2 > 0], lambda x, y: x - y if pd.notnull(x) and
                                      pd.notnull(y) else x) * bridgeFees.groupby('fromChain').bridgeFee.max()
            bridgeFee = bfee.fillna(0).values.sum()

            # Define Gas Fee
            gfee = s1[s1 > 0].combine(s2[s2 > 0], lambda x, y: x + y if pd.notnull(x) and
                                      pd.notnull(y) else x) * gasFees.groupby('chain').gasUsd.max()
            gasFee = gfee.fillna(0).values.sum() / investment

            # Define Swap Fee
            tokensUsed = pd.Series(dtype='float64')
            previousTokensUsed = pd.Series(dtype='float64')
            for ix in range(bestPools.shape[0]):
                tokensUsed = tokensUsed.add(pd.Series(data=bestPools.iloc[ix].position /
                                                      len(bestPools.iloc[ix].token.split("-")), 
                                                      index=bestPools.iloc[ix].token.split("-")), fill_value=0)
            for ix in range(bestPools.shape[0]):
                previousTokensUsed = previousTokensUsed.add(pd.Series(data=previousBestPools.iloc[ix].position /
                                                                      len(previousBestPools.iloc[ix].token.split("-")), 
                                                                      index=previousBestPools.iloc[ix].token.split("-")), fill_value=0)
            swapFee = previousTokensUsed.sub(tokensUsed, fill_value=0).abs().sum()/2 * swapFees

            df.at[df.index[i], 'bridgeFee'] = bridgeFee
            df.at[df.index[i], 'gasFee'] = gasFee
            df.at[df.index[i], 'swapFee'] = swapFee

            base = createDataset(bestPools)
            bestPools['model'] = lgbmodel.predict(base)
            modelScore = (bestPools.position * bestPools.model).sum()
            df.loc[df.index[i], 'modelScore'] = modelScore

            if (yield_upside * yield_upside_factor * modelScore) > (swapFee + bridgeFee + gasFee):
                # If: yield_upside > cost of swaping, bridging & gas
                df.at[df.index[i], 'real_apy'] = df.at[df.index[i], 'max_apy'] - bridgeFee - gasFee - swapFee
                previousBestPools = bestPools
                investment *= (1 + df.at[df.index[i], 'real_apy'])
                df.loc[df.index[i], 'date_of_change'] = df.index[i]

            else:
                # If not, keep the previous pool
                df.loc[df.index[i], 'real_apy'] = currentApr

    # Cummulative result for NOYA
    df['NoyaStrategy_cum'] = df['real_apy'].cumsum()

    # Chain / Protocol Usage
    poolUsed = pd.DataFrame()

    for ix in range(maxPoolsSplit):
        p = df.loc[:,[f'bestPool_{ix+1}', f'position_{ix+1}']]
        p.columns = ['bestPool', 'position']
        poolUsed = pd.concat([poolUsed, p])

    otherTH = 0.03

    poolUsed[['chain', 'protocol', 'token']] = poolUsed.bestPool.str.split('_', expand=True)
    chainUsedPerc = poolUsed.groupby('chain').position.sum() / poolUsed.position.sum()
    chainUsedPerc = pd.concat([chainUsedPerc, pd.Series(chainUsedPerc.loc[chainUsedPerc < otherTH].sum(),['others'])])
    chainUsedPerc = chainUsedPerc.loc[(chainUsedPerc >= otherTH) | (chainUsedPerc.index == 'others')]

    projectUsedPerc = poolUsed.groupby('protocol').position.sum() / poolUsed.position.sum()
    projectUsedPerc = pd.concat([projectUsedPerc, pd.Series(projectUsedPerc.loc[projectUsedPerc < otherTH].sum(),['others'])])
    projectUsedPerc = projectUsedPerc.loc[(projectUsedPerc >= otherTH) | (projectUsedPerc.index == 'others')]

    tokenUsedPerc = poolUsed.groupby('token').position.sum() / poolUsed.position.sum()
    tokenUsedPerc = pd.concat([tokenUsedPerc, pd.Series(tokenUsedPerc.loc[tokenUsedPerc < otherTH].sum(),['others'])])
    tokenUsedPerc = tokenUsedPerc.loc[(tokenUsedPerc >= otherTH) | (tokenUsedPerc.index == 'others')]



    # Cummulative result for Chain / Protocol
    if printAll is True:
        for c in chainList:
            for p in projectList:
                df[f'{c}_{p}_cum'] = df.loc[:, df.columns.str.contains(f"apy-{c}_{p}*")].mean(axis=1).cumsum()
    else:
        for cp in (poolUsed.chain + '_' + poolUsed.protocol).unique():
            df[f'{cp}_cum'] = df.loc[:, df.columns.str.contains(f"apy-{cp}*")].mean(axis=1).cumsum()

    ax = df.loc[:, df.columns.str.contains('_cum')].rename(columns=lambda x: x[:-4]).plot(figsize=(10, 6))

    # Plot Cummulative APR and Pie Charts
    ax.get_lines()[0].set_linewidth(3)
    ax.get_lines()[0].set_color('green')

    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1), fontsize=8, ncol=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # creates 1 row and 3 columns of subplots

    # plot the first pie chart
    axes[0].pie(chainUsedPerc.values, labels=chainUsedPerc.index,
                autopct='%1.0f%%', textprops={'fontsize': 7}, labeldistance=1.2)
    axes[0].set_title('What Blockchains did the strategy use?', fontsize=10)

    # plot the second pie chart
    axes[1].pie(projectUsedPerc.values, labels=projectUsedPerc.index,
                autopct='%1.0f%%', textprops={'fontsize': 7}, labeldistance=1.1)
    axes[1].set_title('What protocols did the strategy use?', fontsize=10)

    # plot the third pie chart
    axes[2].pie(tokenUsedPerc.values, labels=tokenUsedPerc.index,
                autopct='%1.0f%%', textprops={'fontsize': 7})
    axes[2].set_title('What Token did the strategy use?', fontsize=10)

    try:
        print("Noya Cumm Strategy: {:.2%}".format(df.NoyaStrategy_cum[-1]))
        print(f'Portfolio Changes: {df.date_of_change.unique().shape[0]}')
        print(f'Avg Days to change: {round(df.shape[0]/df.date_of_change.unique().shape[0], 1)}')
        print(f'Max Days to change: {np.max(np.diff(df.date_of_change.unique()[1:])).days}')
    except:
        0

    position = previousBestPools.loc[:,['position','chain','protocol','token','tlv','newApr','model']]
    position.rename(columns={"position": "share", "newApr": "expectedAPR", "model": "modelPrediction"}, inplace=True)
    position['expectedAPR'] = (position.expectedAPR * 365 * 100).astype(float).round(2)
    position['modelPrediction'] = (position.modelPrediction * 100).astype(float).round(2)
    position['share'] = (position.share * 100).astype(float).round(2)

    return df, position



##################################################################################################




