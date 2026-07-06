select *
from {{ source('gold_predictions', 'pred_football__player_event_predictions') }}
where predicted_mean is null
   or predicted_mean < 0
   or (predicted_p_ge_1 is not null and (predicted_p_ge_1 < 0 or predicted_p_ge_1 > 1))
   or (predicted_p_ge_2 is not null and (predicted_p_ge_2 < 0 or predicted_p_ge_2 > 1))
   or (predicted_p_ge_3 is not null and (predicted_p_ge_3 < 0 or predicted_p_ge_3 > 1))
   or (predicted_p_ge_1 is not null and predicted_p_ge_2 is not null and predicted_p_ge_2 > predicted_p_ge_1)
   or (predicted_p_ge_2 is not null and predicted_p_ge_3 is not null and predicted_p_ge_3 > predicted_p_ge_2)
