#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm.auto import tqdm


# In[2]:


from Bio import Entrez, SeqIO, Seq


# In[3]:


Entrez.email = "tdn47@drexel.edu"


# # Problem 1:
# How many results do you get when you search for all 28S rRNA genes that are over 700 bp from:

# ## A veterbrate (Hypostomus asperatus)

# In[4]:


handle = Entrez.esearch(
    db="nucleotide",
    term="28S rRNA[gene] AND \
                            Hypostomus asperatus[ORGN] AND \
                            700:9999999999[Sequence Length]",
)
records = Entrez.read(handle)


# In[5]:


print(f"Number of results Hypostomus asperatus: {records['Count']}")


# ## A red algae spieces (Grateloupia turuturu)

# In[6]:


handle = Entrez.esearch(
    db="nucleotide",
    term="28S rRNA[gene] AND \
                            Grateloupia turuturu[ORGN] AND \
                            700:9999999999[Sequence Length]",
)
records = Entrez.read(handle)


# In[7]:


print(f"Number of results Grateloupia turuturu: {records['Count']}")


# ## An basidiomycete fungus (Tricholomopsis flammula)

# In[8]:


handle = Entrez.esearch(
    db="nucleotide",
    term="28S rRNA[gene] AND \
                            Tricholomopsis flammula[ORGN] AND \
                            700:9999999999[Sequence Length] NOT \
                            uncultured",
)
records = Entrez.read(handle)
print(f"Number of results for Tricholomopsis flammula: {records['Count']}")


# # Problem 2:
# Save all 28S rRNA genes that are over 700 bp from all green algae species to a file named “long_28rrna_greenalgae.fa”

# In[9]:


handle_search = Entrez.esearch(
    db="nucleotide",
    term="28S rRNA[gene] AND \
                                    green algae[ORGN] AND \
                                    700:9999999999[Sequence Length] NOT \
                                    uncultured NOT \
                                    partial",
    retmax=1000,
)
records = Entrez.read(handle_search)
print(f"Number of results for green algae: {records['Count']}")
gi_list = records["IdList"]


# In[10]:


f_name = "long_28rrna_greenalgae.fa"
print("Writing to {f_name}")
with open(f_name, "w") as output_file:
    for gi in tqdm(gi_list):
        handle_text = Entrez.efetch(
            db="nucleotide", id=gi, rettype="gb", retmode="text"
        )
        rec_text = SeqIO.read(handle_text, "genbank")
        rec_id = rec_text.id
        rec_desc = rec_text.description

        feature_list = [
            f for f in rec_text.features if f.type == "gene" or f.type == "rRNA"
        ]
        feature_list = [
            f
            for f in feature_list
            if "gene" in f.qualifiers and f.qualifiers["gene"][0] == "28S rRNA"
        ]

        for f in feature_list:
            output_file.write(
                f">{rec_id} {rec_desc} {f.location}\n{f.location.extract(rec_text).seq}\n"
            )


# # Problem 3:
# Save all the Protein IDs of ATPase genes from the first Galdieria sulphuraria whole genome scaffold that you find.  Save them into a file called G_sulphuraria_atpase_ids.

# In[11]:


handle_search = Entrez.esearch(
    db="nucleotide",
    term="Galdieria sulphuraria[ORGN] AND \
                                    (whole OR complete) AND \
                                    scaffold AND \
                                    400000:9999999999[Sequence Length] ",
    retmax=1000,
)
records = Entrez.read(handle_search)
print(f"Number of results for Galdieria sulphuraria: {records['Count']}")
gi = records["IdList"][0]


# In[12]:


f_name = "G_sulphuraria_atpase_ids"
print(f"Writing to {f_name}")
with open(f_name, "w") as output_file:
    handle_text = Entrez.efetch(db="nucleotide", id=gi, rettype="gb", retmode="text")
    rec_text = SeqIO.read(handle_text, "genbank")
    rec_id = rec_text.id
    rec_desc = rec_text.description

    feature_list = [f for f in rec_text.features if f.type == "CDS"]
    feature_list = [
        f
        for f in feature_list
        if "product" in f.qualifiers and "ATPase" in f.qualifiers["product"][0]
    ]

    for f in tqdm(feature_list):
        output_file.write(f"{f.qualifiers['protein_id'][0]}\n")


# In[13]:


handle_search.close()
handle_text.close()


# In[ ]:


# In[ ]:
