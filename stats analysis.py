import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

fortis=pd.read_csv('fortis.csv')
scandens=pd.read_csv('scandens.csv')

#scandens contains parent and offspring beak data
#create appropriate callable series for numpy 

bd_offspring_scandens=scandens['mid_offspring']
bd_parent_scandens=scandens['mid_parent']

bd_offspring_fortis=fortis['Mid-offspr']
bd_parent_fortis = (fortis['Male BD'].values + fortis['Female BD'].values)/2

#Create EDA of heritability of beak depth, compare

# Make scatter plots
_ = sns.set()
_ = plt.subplots(figsize=(10,4))
_ = plt.plot(bd_parent_fortis, bd_offspring_fortis,
             marker='.', linestyle='none', color='blue', alpha=0.5)
             
_ = plt.plot(bd_parent_scandens, bd_offspring_scandens,
             marker='.', linestyle='none', color='red', alpha=0.5)


# Label axes
_ = plt.xlabel('parental beak depth (mm)')
_ = plt.ylabel('offspring beak depth (mm)')

# Add legend
_ = plt.legend(('G. fortis', 'G. scandens'), loc='lower left')

# Show plot
plt.show()

#We can see a stronger correlation in fortis than in scandens. Quantify via
#pearson-r and bootstrapping next

#create functions to draw bootstrap replicates and calculate pearson_r

def draw_bs_pairs(x, y, func, size=1):
    """Draw pairs bootstrap for estimates for params derived via lin. regression"""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates
    bs_replicates=np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds =  np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds],y[bs_inds]
        bs_replicates[i]=func(bs_x,bs_y)

    return bs_replicates

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat=np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]
  
# Compute the Pearson correlation coefficients
r_fortis = pearson_r(bd_parent_fortis, bd_offspring_fortis)
r_scandens = pearson_r(bd_parent_scandens, bd_offspring_scandens)

# Acquire 1000 bootstrap replicates of Pearson r
bs_replicates_scandens = draw_bs_pairs(bd_parent_scandens,bd_offspring_scandens, pearson_r, 1000)
bs_replicates_fortis = draw_bs_pairs(bd_parent_fortis, bd_offspring_fortis, pearson_r, 1000)

#Compute confidence intervals, 95%

conf_int_scandens = np.percentile(bs_replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(bs_replicates_fortis,[2.5, 97.5])

#Show results of pearsons_r

print('G. scandens pearsons_r:', r_scandens, conf_int_scandens)
print('G. fortis pearsons_r:', r_fortis, conf_int_fortis)

#stronger correlation shown in the fortis species. This indicates that 
#the beak depth of fortis parents correlates more strongly to their offspring
#than scandens parents and offspring

#measure heritability
#covariance matrix has variance of parents trait in 0,0 and covariance in 0,1 
#or 1,0.
def heritability(parents, offspring):
    """Compute the heritability from parent and offspring samples."""
    covariance_matrix = np.cov(parents, offspring)
    return covariance_matrix[0,1] / covariance_matrix[0,0]

# Compute the heritability
heritability_scandens = heritability(bd_parent_scandens, bd_offspring_scandens)
heritability_fortis = heritability(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of heritability
replicates_scandens = draw_bs_pairs(
        bd_parent_scandens, bd_offspring_scandens, heritability, size=1000)
        
replicates_fortis = draw_bs_pairs(
        bd_parent_fortis, bd_offspring_fortis, heritability, size=1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(replicates_fortis, [2.5,97.5])

# Print results of heritability
print('G. scandens heritability:', heritability_scandens, conf_int_scandens)
print('G. fortis heritability:', heritability_fortis, conf_int_fortis)

# Initialize array of replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute parent beak depths
    bd_parent_permuted = np.random.permutation(bd_parent_scandens)
    perm_replicates[i] = heritability(bd_parent_permuted, bd_offspring_scandens)


# Compute p-value: p
p = np.sum(perm_replicates >= heritability_scandens) / len(perm_replicates)

# Print the p-value
print('p-val =', p)


#pvalue of 0 means that out of the 10000 trials run, the permutation pairs ]
#replicates drawn had a heritability high enough to match that which was 
#observed. This suggests that beak depth in scandens is heritable, just not as 
#strongly as in fortis.

plt.hist(perm_replicates, bins=20, density=True, stacked=True)
plt.xlabel('Difference in beak depths')
plt.ylabel('PDF')
plt.show()

















