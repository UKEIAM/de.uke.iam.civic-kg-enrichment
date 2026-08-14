#!/usr/bin/env python3
"""Script to fetch CIViC's evidence items via GraphQL and save them to a CSV file.
It filters to only include evidence items with status "accepted" and
keeps relevant fields for downstream processing.

Usage:
    python src/fetch_data/fetch_civic_data.py
"""

import csv
import pathlib
from datetime import datetime, timezone
from pathlib import Path

import requests

GRAPHQL_URL = "https://civicdb.org/api/graphql"
OUTPUT_DIR = Path("data/raw/civic/")

# GraphQL query template
query_template = """
query($first: Int!, $after: String) {
  evidenceItems(status: ACCEPTED, first: $first, after: $after) {
    nodes {
      id
      status
      evidenceType
      evidenceDirection
      significance
      evidenceLevel
      variantOrigin
      therapyInteractionType
      evidenceRating
      description

      source {
          id
          sourceType
          citation
          citationId
          title
          abstract
      }

      disease {
        id
        name
        doid
        diseaseAliases
        myDiseaseInfo {
          mondoId
          mesh
          ncit
          icd10
          diseaseOntologyExactSynonyms
          diseaseOntologyRelatedSynonyms
        }
      }

      therapies {
        id
        name
        ncitId
        therapyAliases
        myChemInfo {
          drugbankId
          rxnorm
          pubchemCid
          chemblId
          chebiId
          inchikey
          pharmgkbId
        }
      }

      molecularProfile {
        id
        name
        isComplex
        isMultiVariant
        variants {
          ... on GeneVariant {
            id
            name
            hgvsDescriptions
            alleleRegistryId
            clinvarIds
            feature {
              id
              name
              featureType
              featureInstance {
                ... on Gene {
                  id
                  name
                  entrezId
                }
              }
            }
            myVariantInfo {
              dbsnpRsid
              clinvarId
              clinvarHgvsCoding
              clinvarHgvsProtein
              clinvarHgvsGenomic
              cosmicId
            }
          }
        }
      }
    }
    pageInfo {
        endCursor
        hasNextPage
    }
  }
}
"""


def main():
    # Pagination loop
    all_items = []
    first = 5000
    after = None

    while True:
        variables = {"first": first, "after": after}
        response = requests.post(
            GRAPHQL_URL, json={"query": query_template, "variables": variables}
        )
        response.raise_for_status()
        data = response.json()

        nodes = data["data"]["evidenceItems"]["nodes"]
        page_info = data["data"]["evidenceItems"]["pageInfo"]

        all_items.extend(nodes)
        print(f"Fetched {len(nodes)} items, total so far: {len(all_items)}")

        if not page_info["hasNextPage"]:
            break
        after = page_info["endCursor"]

    print(f"Finished fetching all evidence items. Total: {len(all_items)}")

    def s(value):
        """Coerce a possibly-None GraphQL scalar to a joinable string."""
        return "" if value is None else str(value)

    def g(d, *keys):
        """Walk nested dicts safely, tolerating missing keys and explicit nulls."""
        for key in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(key)
        return d

    # Flatten items for CSV
    def flatten(item):
        disease = item.get("disease") or {}
        disease_info = disease.get("myDiseaseInfo") or {}
        source = item.get("source") or {}
        therapies = item.get("therapies") or []
        molecular_profile = item.get("molecularProfile") or {}
        variants = molecular_profile.get("variants") or []

        return {
            "id": item.get("id"),
            "status": item.get("status"),
            "evidence_type": item.get("evidenceType"),
            "evidence_direction": item.get("evidenceDirection"),
            "significance": item.get("significance"),
            "evidence_level": item.get("evidenceLevel"),
            "variant_origin": item.get("variantOrigin"),
            "evidence_rating": item.get("evidenceRating"),
            "description": item.get("description"),
            # disease-related fields
            "disease_id": disease.get("id"),
            "disease_name": disease.get("name"),
            "disease_doid": disease.get("doid"),
            "disease_aliases": ",".join(
                s(a) for a in disease.get("diseaseAliases") or []
            ),
            "disease_mondo_id": disease_info.get("mondoId"),
            "disease_mesh": disease_info.get("mesh"),
            "disease_ncit": disease_info.get("ncit"),
            "disease_icd10": disease_info.get("icd10"),
            # therapy-related fields
            "therapy_id": ",".join(s(t.get("id")) for t in therapies),
            "therapy_name": ",".join(s(t.get("name")) for t in therapies),
            "therapy_ncit_id": ",".join(s(t.get("ncitId")) for t in therapies),
            "therapy_aliases": ",".join(
                s(a) for t in therapies for a in t.get("therapyAliases") or []
            ),
            "therapy_drugbank_id": ",".join(
                s(g(t, "myChemInfo", "drugbankId")) for t in therapies
            ),
            "therapy_rxnorm": ",".join(
                s(g(t, "myChemInfo", "rxnorm")) for t in therapies
            ),
            "therapy_pubchem_cid": ",".join(
                s(g(t, "myChemInfo", "pubchemCid")) for t in therapies
            ),
            "therapy_chembl_id": ",".join(
                s(g(t, "myChemInfo", "chemblId")) for t in therapies
            ),
            "therapy_chebi_id": ",".join(
                s(g(t, "myChemInfo", "chebiId")) for t in therapies
            ),
            "therapy_inchikey": ",".join(
                s(g(t, "myChemInfo", "inchikey")) for t in therapies
            ),
            "therapy_pharmgkb_id": ",".join(
                s(g(t, "myChemInfo", "pharmgkbId")) for t in therapies
            ),
            "therapy_interaction_type": item.get("therapyInteractionType"),
            # molecular profile-related fields
            "molecular_profile": molecular_profile.get("name"),
            "molecular_profile_id": molecular_profile.get("id"),
            "molecular_profile_is_complex": molecular_profile.get("isComplex"),
            "molecular_profile_is_multi_variant": molecular_profile.get(
                "isMultiVariant"
            ),
            "variants": ",".join(s(v.get("name")) for v in variants),
            "variant_ids": ",".join(s(v.get("id")) for v in variants),
            "variant_hgvs_descriptions": ",".join(
                ",".join(s(h) for h in v.get("hgvsDescriptions") or [])
                for v in variants
            ),
            "variant_allele_registry_ids": ",".join(
                s(v.get("alleleRegistryId")) for v in variants
            ),
            "variant_clinvar_ids": ",".join(
                ",".join(s(c) for c in v.get("clinvarIds") or []) for v in variants
            ),
            "variant_feature_ids": ",".join(s(g(v, "feature", "id")) for v in variants),
            "variant_feature_names": ",".join(
                s(g(v, "feature", "name")) for v in variants
            ),
            "variant_feature_types": ",".join(
                s(g(v, "feature", "featureType")) for v in variants
            ),
            "variant_feature_instance_ids": ",".join(
                s(g(v, "feature", "featureInstance", "id")) for v in variants
            ),
            "variant_feature_instance_names": ",".join(
                s(g(v, "feature", "featureInstance", "name")) for v in variants
            ),
            "variant_feature_entrez_ids": ",".join(
                s(g(v, "feature", "featureInstance", "entrezId")) for v in variants
            ),
            "variant_dbsnp_rsids": ",".join(
                s(g(v, "myVariantInfo", "dbsnpRsid")) for v in variants
            ),
            "variant_clinvar_id": ",".join(
                s(g(v, "myVariantInfo", "clinvarId")) for v in variants
            ),
            "variant_clinvar_hgvs_coding": ",".join(
                s(g(v, "myVariantInfo", "clinvarHgvsCoding")) for v in variants
            ),
            "variant_clinvar_hgvs_protein": ",".join(
                s(g(v, "myVariantInfo", "clinvarHgvsProtein")) for v in variants
            ),
            "variant_clinvar_hgvs_genomic": ",".join(
                s(g(v, "myVariantInfo", "clinvarHgvsGenomic")) for v in variants
            ),
            "variant_cosmic_ids": ",".join(
                s(g(v, "myVariantInfo", "cosmicId")) for v in variants
            ),
            # source-related fields
            "source_id": source.get("id"),
            "source_type": source.get("sourceType"),
            "source_citation": source.get("citation"),
            "source_citation_id": source.get("citationId"),
        }

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Flatten all items
    flat_items = [flatten(i) for i in all_items]

    # Write to CSV
    fieldnames = list(flat_items[0].keys())

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with open(
        pathlib.Path(OUTPUT_DIR) / f"civic_raw_{date_str}.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_items)

    print(f"✅ Saved raw CIViC data to CSV in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
