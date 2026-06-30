with rankings_seed as (
    select
        cast(`Rank` as int) as rank,
        `Team` as team_name,
        cast(`Raiting` as double) as rating
    from {{ ref('fifa_mens_world_ranking_december_2022') }}
)

select
    rank,
    team_name,
    rating,
    to_date('{{ var("ranking_as_of_date") }}') as ranking_as_of_date,
    {{ normalize_name('team_name') }} as team_name_normalized,
    current_timestamp() as updated_at_utc
from rankings_seed
