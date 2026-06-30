{% macro normalize_name(column_expression) -%}
  trim(regexp_replace(
    regexp_replace(
      translate(lower(trim({{ column_expression }})),
        'ÀÁÂÃÄÅĀĂĄÇĆČÐĎÈÉÊËĒĔĖĘĚÌÍÎÏĪĮİŁÑŃŇÒÓÔÕÖØŌŐŘŚŞŠÙÚÛÜŪŮŰÝŸŽŹŻàáâãäåāăąçćčðďèéêëēĕėęěìíîïīįıłñńňòóôõöøōőřśşšùúûüūůűýÿžźż',
        'AAAAAAAAACCCDDEEEEEEEEEIIIIIIILNNNOOOOOOOORSSSUUUUUUUYYYZZZaaaaaaaaacccddeeeeeeeeeiiiiiilnnnoooooooorsssuuuuuuuyyzzz'
      ),
      '[^a-z0-9]+',
      ' '
    ),
    '\\s+',
    ' '
  ))
{%- endmacro %}
