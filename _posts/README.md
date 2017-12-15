#Project3
***
***Course:*** *Operating System*

***Name:*** *Fei Gao* 
***

##Part1: Small size Files

In this part, I achieve all the requirements.


###`file_create()`
```c
		inode[inodeNum].type = file;
		inode[inodeNum].owner = 1;
		inode[inodeNum].group = 2;
		gettimeofday(&(inode[inodeNum].created), NULL);
		gettimeofday(&(inode[inodeNum].lastAccess), NULL);
		inode[inodeNum].size = size;
		inode[inodeNum].blockCount = numBlock;
		inode[inodeNum].inblockCount = 0;
		
		strncpy(curDir.dentry[curDir.numEntry].name, name, strlen(name));
		curDir.dentry[curDir.numEntry].name[strlen(name)] = '\0';
		curDir.dentry[curDir.numEntry].inode = inodeNum;
		printf("curdir %s, name %s\n", curDir.dentry[curDir.numEntry].name, name);
		curDir.numEntry++;

		for(i = 0; i < numBlock; i++)
		{
				int block = get_free_block();
				if(block == -1) {
						printf("File_create error: get_free_block failed\n");
						return -1;
				}
				inode[inodeNum].directBlock[i] = block;
				disk_write(block, tmp+(i*BLOCK_SIZE));
		}
```
###`file_cat()`
```c
		int numBlock = inode[inodeNum].blockCount;
		gettimeofday(&(inode[inodeNum].lastAccess),NULL);
            
		char str[numBlock*BLOCK_SIZE];
		for(int i = 0; i<numBlock; i++)
		{
			disk_read(inode[inodeNum].directBlock[i],str+i*BLOCK_SIZE);
		}
		printf("%s\n",str);
		return 1;
```
###`file_read()`
```c
		int numBlock = inode[inodeNum].blockCount;
		gettimeofday(&(inode[inodeNum].lastAccess),NULL);

		char buf[numBlock*BLOCK_SIZE];
		char str[size+1];
		for(int i=0; i<numBlock; i++)
		{
			disk_read(inode[inodeNum].directBlock[i],buf+i*BLOCK_SIZE);
		}
		strncpy(str,buf+offset,size);
		str[size] = '\0';
		printf("%s\n",str);
		return 1;
```
###`file_write()`
```c
		gettimeofday(&(inode[inodeNum].lastAccess),NULL);
		int thisBlock = offset/BLOCK_SIZE;
		int offsetBlock = offset%BLOCK_SIZE;
		char tmpbuf[BLOCK_SIZE];
		int curptr=0;

		while(size-(BLOCK_SIZE-offsetBlock)>0)
		{
			disk_read(inode[inodeNum].directBlock[thisBlock], tmpbuf);
			memcpy(tmpbuf+offsetBlock,buf+curptr,BLOCK_SIZE-offsetBlock);
			int block = inode[inodeNum].directBlock[thisBlock];
			disk_write(block, tmpbuf);
			size-=BLOCK_SIZE-offsetBlock;
			curptr+=BLOCK_SIZE-offsetBlock;
			if(offsetBlock!=0)offsetBlock=0;
			thisBlock++;
		}
		disk_read(inode[inodeNum].directBlock[thisBlock], tmpbuf);
		memcpy(tmpbuf+offsetBlock,buf,size);
		int block = inode[inodeNum].directBlock[thisBlock];
		disk_write(block, tmpbuf);
		file_cat(name);
		printf("write complete\n");
```
###`file_remove()`
```c
		for(int i = 0; i < curDir.numEntry; i++)
		{
			if(command(name, curDir.dentry[i].name))
			{
	    		char *endName;
	    		int endInode;
	    		strncpy(endName, curDir.dentry[curDir.numEntry-1].name, strlen(curDir.dentry[curDir.numEntry-1].name));
	    		endInode = curDir.dentry[curDir.numEntry-1].inode;

	    		strncpy(curDir.dentry[i].name, endName, strlen(endName));
	    		curDir.dentry[i].inode = endInode;
	    		curDir.dentry[curDir.numEntry-1].inode = 0;
	    		curDir.numEntry--;
	    		curDir.dentry[curDir.numEntry-1].name[strlen(curDir.dentry[curDir.numEntry-1].name)] = '\0';

	    	}
	    }
	    
		for(int i = 0; i < inode[inodeNum].blockCount; i++)
		{
			set_bit(blockMap, inode[inodeNum].directBlock[i],0);
			superBlock.freeBlockCount++;
		}
		set_bit(inodeMap, inodeNum, 0);
		superBlock.freeInodeCount++;
		return 1;
```
##Part2: Directory
In this part, I achieve all the requirements except rmdir.
###`dir_make()`
```c
	 	strncpy(curDir.dentry[curDir.numEntry].name, name, strlen(name));
	 	curDir.dentry[curDir.numEntry].name[strlen(name)] = '\0';
	 	curDir.dentry[curDir.numEntry].inode = inodeNum;
	 	curDir.numEntry++;
	 	inode[inodeNum].type = directory;
	 	inode[inodeNum].owner = 1;
	 	inode[inodeNum].group = 2;
	 	gettimeofday(&(inode[inodeNum].created),NULL);
	 	gettimeofday(&(inode[inodeNum].lastAccess),NULL);
	 	inode[inodeNum].size = 1;
	 	inode[inodeNum].blockCount = 1;
	 	inode[inodeNum].inblockCount = 0;
	 	inode[inodeNum].directBlock[0] = dBlock;

	 	Dentry newDir;
	 	newDir.numEntry = 2;
	 	strncpy(newDir.dentry[0].name, ".", 1);
	 	strncpy(newDir.dentry[1].name, "..", 2);
	 	newDir.dentry[0].name[1] = '\0';
	 	newDir.dentry[0].inode = inodeNum;
	 	newDir.dentry[1].name[2] = '\0';
	 	newDir.dentry[1].inode = curDir.dentry[0].inode;
	 	disk_write(dBlock, (char*)&newDir);
	 	int pBlock = inode[curDir.dentry[0].inode].directBlock[0];
	 	disk_write(pBlock, (char*)&curDir);
		 return 0;
```
###`dir_change()`
```c
		curDirBlock = inode[inodeNum].directBlock[0];
		disk_read(curDirBlock,(char*)&curDir);
		return 1;
```
##Part3: Large size file
In this part, I use `indirectBlock` to record the No. of blocks on disk.

In `file_create()`, using 10 directBlocks to record the first 10 No. of blocks, and using indirectBlocks to record the rest No. of blocks.

In`file_cat()`, reading the content from 10 directBlocks and all indirectBlocks.

In`file_read()`, same as `file_cat()`, recording the content from 10 directBlocks and all indirectBlocks first, then according *offset* and *size* to read.

In`file_remove()`, free all the directBlocks and indirectBlocks, also update metadata.

##Result

- `ls` to show the current directory
- `create fileA 5` to create a file with size 5
- `cat fileA` to read the content of fileA
- `read fileA 3 2` to read 2 bytes from offset 3
- `write fileA 3 2 12` to overwrite 2 bytes "12" in fileA from offset 3 
- `mkdir dir1` to create a directory
- `cd dir1` to enter the dir1
- `ls` to show the dir1
- `create fileLarge 7000` to create a large file
![avatar](/Users/gavin/Desktop/1.png)

- `read fileLarge 6998 2` to read 2 bytes from offset 6998
- `cd ..` to go back to parent dir
- `rm fileA` to remove a file


![avatar](/Users/gavin/Desktop/2.png)
