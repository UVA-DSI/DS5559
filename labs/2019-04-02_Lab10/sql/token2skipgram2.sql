WITH mytoken(author, book, chapter, para_num, sent_num,token_num,term_str,term_id) AS (
	SELECT author, book, chapter, para_num, sent_num,token_num,term_str,term_id
	FROM  token WHERE term_id IN (SELECT term_id FROM vocab WHERE stop = 0)
	AND term_str is not NULL
)
select x.term_str as core, y.term_str as partner, (y.token_num - x.token_num) as dist
from mytoken x 
	join mytoken y using(author, book, chapter, para_num, sent_num)
where 
    y.token_num in (
		x.token_num - 2, x.token_num - 1, 
		x.token_num + 1, x.token_num + 2 
	)
order  by core, dist, partner