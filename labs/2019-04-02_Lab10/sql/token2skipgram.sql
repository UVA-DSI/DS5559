select x.term_str as core, y.term_str as partner
from token x join token y using(author, book, chapter, para_num, sent_num)
where y.token_num in (x.token_num +1, y.token_num = x.token_num +2,
	y.token_num = x.token_num -1, y.token_num = x.token_num -2)
	and x.term_str is not NULL
	and y.term_str is not NULL
	
	