# Databricks notebook source
from football_analytics.databricks_ingestion import natural_key_merge_plans

plans = natural_key_merge_plans()
display([(name, plan.table, ", ".join(plan.keys), plan.predicate) for name, plan in plans.items()])

